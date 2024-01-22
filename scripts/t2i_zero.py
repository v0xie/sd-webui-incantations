import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import scipy.stats as stats

from scripts.ui_wrapper import UIWrapper, arg
from modules import script_callbacks, prompt_parser, sd_hijack, sd_hijack_optimizations
from modules.hypernetworks import hypernetwork
#import modules.sd_hijack_optimizations
from modules.script_callbacks import CFGDenoiserParams
from modules.prompt_parser import reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import pad_cond
from modules import shared

import math
import torch
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur

from warnings import warn
from typing import Callable, Dict, Optional
from collections import OrderedDict
import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""

Unofficial implementation of algorithms in Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else

@misc{tunanyan2023multiconcept,
      title={Multi-Concept T2I-Zero: Tweaking Only The Text Embeddings and Nothing Else}, 
      author={Hazarapet Tunanyan and Dejia Xu and Shant Navasardyan and Zhangyang Wang and Humphrey Shi},
      year={2023},
      eprint={2310.07419},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-semantic-guidance

"""

handles = []

class SegaStateParams:
        def __init__(self):
                self.concept_name = ''
                self.v = {} # velocity
                self.warmup_period: int = 10 # [0, 20]
                self.tail_percentage_threshold: float = 0.05 # [0., 1.] if abs value of difference between uncodition and concept-conditioned is less than this, then zero out the concept-conditioned values less than this
                self.momentum_scale: float = 0.3 # [0., 1.]
                self.momentum_beta: float = 0.6 # [0., 1.) # larger bm is less volatile changes in momentum
                self.strength = 1.0
                self.width = None
                self.height = None
                self.dims = []

class SegaExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "Multi T2I-Zero"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Multi T2I-Zero', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='t2i0_active')
                        with gr.Row():
                                warmup = gr.Slider(value = 10, minimum = 0, maximum = 30, step = 1, label="Window Size", elem_id = 't2i0_warmup', info="How many steps to wait before applying semantic guidance, default 10")
                                momentum_scale = gr.Slider(value = 0.3, minimum = 0.0, maximum = 1.0, step = 0.01, label="Correction Threshold", elem_id = 't2i0_momentum_scale', info="Scale of momentum, default 0.3")
                                momentum_beta = gr.Slider(value = 0.6, minimum = 0.0, maximum = 0.999, step = 0.01, label="Correction Strength", elem_id = 't2i0_momentum_beta', info="Beta for momentum, default 0.6")
                        with gr.Row():
                                tail_percentage_threshold = gr.Slider(value = 0.05, minimum = 0.0, maximum = 1.0, step = 0.01, label="Alpha for CTNMS", elem_id = 't2i0_tail_percentage_threshold', info="The percentage of latents to modify, default 0.05")
                active.do_not_save_to_config = True
                warmup.do_not_save_to_config = True
                tail_percentage_threshold.do_not_save_to_config = True
                momentum_scale.do_not_save_to_config = True
                momentum_beta.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='T2I-0 Active' in d)),
                        (warmup, 'T2I-0 Warmup Period'),
                        (tail_percentage_threshold, 'T2I-0 Tail Percentage Threshold'),
                        (momentum_scale, 'T2I-0 Momentum Scale'),
                        (momentum_beta, 'T2I-0 Momentum Beta'),
                ]
                self.paste_field_names = [
                        't2i0_active',
                        't2i0_prompt',
                        't2i0_neg_prompt',
                        't2i0_warmup',
                        't2i0_tail_percentage_threshold',
                        't2i0_momentum_scale',
                        't2i0_momentum_beta'
                ]
                return [active, warmup, tail_percentage_threshold, momentum_scale, momentum_beta]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.sega_process_batch(p, *args, **kwargs)

        def sega_process_batch(self, p: StableDiffusionProcessing, active, warmup, tail_percentage_threshold, momentum_scale, momentum_beta, *args, **kwargs):
                active = getattr(p, "t2i0_active", active)
                if active is False:
                        return
                warmup = getattr(p, "t2i0_warmup", warmup)
                tail_percentage_threshold = getattr(p, "t2i0_tail_percentage_threshold", tail_percentage_threshold)
                momentum_scale = getattr(p, "t2i0_momentum_scale", momentum_scale)
                momentum_beta = getattr(p, "t2i0_momentum_beta", momentum_beta)
                p.extra_generation_params.update({
                        "T2I-0 Active": active,
                        "T2I-0 Warmup Period": warmup,
                        "T2I-0 Tail Percentage Threshold": tail_percentage_threshold,
                        "T2I-0 Momentum Scale": momentum_scale,
                        "T2I-0 Momentum Beta": momentum_beta,
                })

                self.create_hook(p, active, warmup, tail_percentage_threshold, momentum_scale, momentum_beta, p.width, p.height)

        def parse_concept_prompt(self, prompt:str) -> list[str]:
                """
                Separate prompt by comma into a list of concepts
                TODO: parse prompt into a list of concepts using A1111 functions
                >>> g = lambda prompt: self.parse_concept_prompt(prompt)
                >>> g("")
                []
                >>> g("apples")
                ['apples']
                >>> g("apple, banana, carrot")
                ['apple', 'banana', 'carrot']
                """
                if len(prompt) == 0:
                        return []
                return [x.strip() for x in prompt.split(",")]

        def create_hook(self, p, active, warmup, tail_percentage_threshold, momentum_scale, momentum_beta, width, height, *args, **kwargs):
                # Create a list of parameters for each concept
                concepts_sega_params = []

                #for _, strength in concept_conds:
                sega_params = SegaStateParams()
                sega_params.warmup_period = warmup
                sega_params.tail_percentage_threshold = tail_percentage_threshold
                sega_params.momentum_scale = momentum_scale
                sega_params.momentum_beta = momentum_beta
                sega_params.strength = 1.0
                sega_params.width = width
                sega_params.height = height 
                sega_params.dims = [width, height]
                concepts_sega_params.append(sega_params)

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, concepts_sega_params)
                un = lambda params: self.unhook_callbacks()

                # Hook callbacks
                if tail_percentage_threshold > 0:
                        self.ready_hijack_forward(sega_params, tail_percentage_threshold, width, height)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, *args, **kwargs):
                self.sega_postprocess_batch(p, *args, **kwargs)

        def sega_postprocess_batch(self, p, active, neg_text, *args, **kwargs):
                self.unhook_callbacks()
                active = getattr(p, "sega_active", active)
                if active is False:
                        return

        def unhook_callbacks(self):
                global handles
                logger.debug('Unhooked callbacks')
                cross_attn_modules = self.get_cross_attn_modules()
                for module in cross_attn_modules:
                        _remove_all_forward_hooks(module, 'cross_token_non_maximum_suppression')
                script_callbacks.remove_current_script_callbacks()

        def correction_by_similarities(self, f, C, percentile, gamma, alpha):
                """
                Apply the Correction by Similarities algorithm on embeddings.

                Args:
                f (Tensor): The embedding tensor of shape (n, d).
                C (list): Indices of selected tokens.
                percentile (float): Percentile to use for score threshold.
                gamma (int): Window size for the windowing function.
                alpha (float): Correction strength.

                Returns:
                Tensor: The corrected embedding tensor.
                """
                if alpha == 0:
                        return f

                n, d = f.shape
                f_tilde = f.detach().clone()  # Copy the embedding tensor

                # Define a windowing function
                def psi(c, gamma, n, dtype, device):
                        window = torch.zeros(n, dtype=dtype, device=device)
                        start = max(0, c - gamma)
                        end = min(n, c + gamma + 1)
                        window[start:end] = 1
                        return window

                for token_idx, c in enumerate(C):
                        Sc = f[c] * f  # Element-wise multiplication
                        # product = greater positive value indicates more similarity
                        # filter out values under score threshold from 0 to max
                        Sc_flat_positive = Sc[Sc > 0]
                        k = 10
                        e= 2.718281
                        # 0.000001 < pct < 0.999999999
                        pct = min(0.999999999, max(0.000001, 1 - e**(-k * percentile))) 
                        tau = torch.quantile(Sc_flat_positive, pct)
                        Sc_tilde = Sc * (Sc > tau)  # Apply threshold and filter
                        Sc_tilde /= Sc_tilde.max()  # Normalize
                        window = psi(c, gamma, n, Sc_tilde.dtype, Sc_tilde.device).unsqueeze(1)  # Apply windowing function
                        Sc_tilde *= window
                        f_c_tilde = torch.sum(Sc_tilde * f, dim=0)  # Combine embeddings
                        f_tilde[c] = (1 - alpha) * f[c] + alpha * f_c_tilde  # Blend embeddings

                return f_tilde

        def ready_hijack_forward(self, sega_params, alpha, width, height):
                cross_attn_modules = self.get_cross_attn_modules()

                def cross_token_non_maximum_suppression(module, input, kwargs, output):
                        context = kwargs.get('context', None)
                        if context is None:
                                return
                        batch_size, sequence_length, inner_dim = output.shape
                        #print(f"\nBatch size: {batch_size}, sequence length: {sequence_length}, inner dim: {inner_dim}\n")

                        max_dims = width*height
                        factor = math.isqrt(max_dims // sequence_length) # should be a square of 2
                        downscale_width = width // factor
                        downscale_height = height // factor
                        if downscale_width * downscale_height != sequence_length:
                                print(f"Error: Width: {width}, height: {height}, Downscale width: {downscale_width}, height: {downscale_height}, Factor: {factor}, Max dims: {max_dims}\n")
                                return

                        h = module.heads
                        head_dim = inner_dim // h
                        dtype = output.dtype
                        device = output.device

                        # Reshape the attention map to batch_size, height, width
                        # FIXME: need to assert the height/width divides into the sequence length
                        attention_map = output.view(batch_size, downscale_height, downscale_width, inner_dim)
                        # attention_map = output.view(batch_size, downscale_height, downscale_width, h, head_dim)

                        # Select token indices (Assuming this is provided as sega_params or similar)
                        selected_tokens = torch.tensor(list(range(inner_dim)))  # Example: Replace with actual indices

                        # Extract and process the selected attention maps
                        # GaussianBlur expects the input [..., C, H, W]
                        gaussian_blur = GaussianBlur(kernel_size=3, sigma=1)
                        AC = attention_map[:, :, :, selected_tokens]  # Extracting relevant attention maps
                        AC = AC.permute(0, 3, 1, 2)
                        AC = gaussian_blur(AC)  # Applying Gaussian smoothing
                        AC = AC.permute(0, 2, 3, 1)

                        # Find the maximum contributing token for each pixel
                        M = torch.argmax(AC, dim=-1)

                        # Create one-hot vectors for suppression
                        t = attention_map.size(-1)
                        one_hot_M = F.one_hot(M, num_classes=t).to(dtype=dtype, device=device)

                        # Apply the suppression mask
                        #suppressed_attention_map = one_hot_M.unsqueeze(2) * attention_map
                        suppressed_attention_map = one_hot_M * attention_map

                        # Reshape back to original dimensions
                        suppressed_attention_map = suppressed_attention_map.view(batch_size, sequence_length, inner_dim)

                        out_tensor = (1-alpha) * output + alpha * suppressed_attention_map

                        return out_tensor

                for module in cross_attn_modules:
                        handle = module.register_forward_hook(cross_token_non_maximum_suppression, with_kwargs=True)

        def get_cross_attn_modules(self):
            m = shared.sd_model
            nlm = m.network_layer_mapping
            cross_attn_modules = [m for m in nlm.values() if 'CrossAttention' in m.__class__.__name__]
            return cross_attn_modules

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, sega_params: list[SegaStateParams]):
                # TODO: add option to opt out of batching for performance
                sampling_step = params.sampling_step

                # SDXL
                if isinstance(params.text_cond, dict):
                        text_cond = params.text_cond['crossattn']
                # SD 1.5
                else:
                        text_cond = params.text_cond


                sp = sega_params[0]
                window_size = sp.warmup_period
                correction_strength = sp.momentum_beta
                score_threshold = sp.momentum_scale
                width = sp.width
                height = sp.height
                tail_percentage_threshold = sp.tail_percentage_threshold

                for batch_idx, batch in enumerate(text_cond):
                        window = list(range(0, len(batch)))

                        f_bar = self.correction_by_similarities(batch, window, score_threshold, window_size, correction_strength)

                        if isinstance(params.text_cond, dict):
                                params.text_cond['crossattn'][batch_idx] = f_bar
                        else:
                                params.text_cond[batch_idx] = f_bar
                return

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[T2I-0] Active", str, sega_apply_override('t2i0_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[T2I-0] Tail Percentage Threshold", float, sega_apply_field("t2i0_tail_percentage_threshold")),
                        xyz_grid.AxisOption("[T2I-0] Window Size", int, sega_apply_field("t2i0_warmup")),
                        xyz_grid.AxisOption("[T2I-0] Correction Threshold", float, sega_apply_field("t2i0_momentum_scale")),
                        xyz_grid.AxisOption("[T2I-0] Correction Strength", float, sega_apply_field("t2i0_momentum_beta")),
                }
                return extra_axis_options

# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def sega_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def sega_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "t2i0_active"):
                setattr(p, "t2i0_active", True)
        setattr(p, field, x)
    return fun


# removing hooks DOESN'T WORK
# https://github.com/pytorch/pytorch/issues/70455
def _remove_all_forward_hooks(
    module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> None:
    """
    This function removes all forward hooks in the specified module, without requiring
    any hook handles. This lets us clean up & remove any hooks that weren't property
    deleted.

    Warning: Various PyTorch modules and systems make use of hooks, and thus extreme
    caution should be exercised when removing all hooks. Users are recommended to give
    their hook function a unique name that can be used to safely identify and remove
    the target forward hooks.

    Args:

        module (nn.Module): The module instance to remove forward hooks from.
        hook_fn_name (str, optional): Optionally only remove specific forward hooks
            based on their function's __name__ attribute.
            Default: None
    """

    if hook_fn_name is None:
        warn("Removing all active hooks can break some PyTorch modules & systems.")


    def _remove_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(module, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                if name is not None:
                    dict_items = list(m._forward_hooks.items())
                    m._forward_hooks = OrderedDict(
                        [(i, fn) for i, fn in dict_items if fn.__name__ != name]
                    )
                else:
                    m._forward_hooks: Dict[int, Callable] = OrderedDict()

    def _remove_child_hooks(
        target_module: torch.nn.Module, hook_name: Optional[str] = None
    ) -> None:
        for name, child in target_module._modules.items():
            if child is not None:
                _remove_hooks(child, hook_name)
                _remove_child_hooks(child, hook_name)

    # Remove hooks from target submodules
    _remove_child_hooks(module, hook_fn_name)

    # Remove hooks from the target module
    _remove_hooks(module, hook_fn_name)