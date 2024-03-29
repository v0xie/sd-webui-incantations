import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import scipy.stats as stats

from scripts.ui_wrapper import UIWrapper, arg
from modules import script_callbacks
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
PAG perturbed attention guidance
todo: citation

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""

handles = []

class PAGStateParams:
        def __init__(self):
                self.attnreg: bool = False
                self.ema_smoothing_factor: float = 2.0
                self.step_end : int = 25
                self.tokens: str = "" # [0, 20]
                self.window_size_period: int = 10 # [0, 20]
                self.ctnms_alpha: float = 0.05 # [0., 1.] if abs value of difference between uncodition and concept-conditioned is less than this, then zero out the concept-conditioned values less than this
                self.correction_threshold: float = 0.5 # [0., 1.]
                self.correction_strength: float = 0.25 # [0., 1.) # larger bm is less volatile changes in momentum
                self.strength = 1.0
                self.width = None
                self.height = None
                self.dims = []

class PAGExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "PAG"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('PAG', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='pag_active')
                        step_end = gr.Slider(value=25, minimum=0, maximum=150, default=1, step=1, label="Step End", elem_id='pag_step_end')
                        with gr.Row():
                                tokens = gr.Textbox(visible=False, value="", label="Tokens", elem_id='pag_tokens', info="Comma separated list of tokens to condition on")
                        with gr.Row():
                                window_size = gr.Slider(value = 3, minimum = 0, maximum = 100, step = 1, label="Correction by Similarities Window Size", elem_id = 'pag_window_size', info="Exclude contribution of tokens further than this from the current token")
                                correction_threshold = gr.Slider(value = 0.0, minimum = 0., maximum = 1.0, step = 0.001, label="CbS Score Threshold", elem_id = 'pag_correction_threshold', info="Filter dimensions with similarity below this threshold")
                                correction_strength = gr.Slider(value = 0.0, minimum = 0.0, maximum = 0.999, step = 0.01, label="CbS Correction Strength", elem_id = 'pag_correction_strength', info="The strength of the correction, default 0.1")
                        with gr.Row():
                                attnreg = gr.Checkbox(visible=False, value=False, default=False, label="Use Attention Regulation", elem_id='pag_use_attnreg')
                                ctnms_alpha = gr.Slider(value = 0.1, minimum = 0.0, maximum = 1.0, step = 0.01, label="Alpha for Cross-Token Non-Maximum Suppression", elem_id = 'pag_ctnms_alpha', info="Contribution of the suppressed attention map, default 0.1")
                                ema_factor = gr.Slider(value=0.0, minimum=0.0, maximum=4.0, default=2.0, label="EMA Smoothing Factor", elem_id='pag_ema_factor')
                active.do_not_save_to_config = True
                attnreg.do_not_save_to_config = True
                ema_factor.do_not_save_to_config = True
                step_end.do_not_save_to_config = True
                window_size.do_not_save_to_config = True
                ctnms_alpha.do_not_save_to_config = True
                correction_threshold.do_not_save_to_config = True
                correction_strength.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='PAG Active' in d)),
                        #(attnreg, lambda d: gr.Checkbox.update(value='PAG AttnReg' in d)),
                        (step_end, 'PAG Step End'),
                        (window_size, 'PAG Window Size'),
                        (ctnms_alpha, 'PAG CTNMS Alpha'),
                        (correction_threshold, 'PAG CbS Score Threshold'),
                        (correction_strength, 'PAG CbS Correction Strength'),
                        (ema_factor, 'PAG CTNMS EMA Smoothing Factor'),
                ]
                self.paste_field_names = [
                        'pag_active',
                        'pag_attnreg',
                        'pag_window_size',
                        'pag_ctnms_alpha',
                        'pag_correction_threshold',
                        'pag_correction_strength'
                        'pag_ema_factor',
                        'pag_step_end'
                ]
                return [active, attnreg, window_size, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.pag_process_batch(p, *args, **kwargs)

        def pag_process_batch(self, p: StableDiffusionProcessing, active, attnreg, window_size, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end, *args, **kwargs):
                active = getattr(p, "pag_active", active)
                use_attnreg = getattr(p, "pag_attnreg", attnreg)
                ema_factor = getattr(p, "pag_ema_factor", ema_factor)
                step_end = getattr(p, "pag_step_end", step_end)
                if active is False:
                        return
                window_size = getattr(p, "pag_window_size", window_size)
                ctnms_alpha = getattr(p, "pag_ctnms_alpha", ctnms_alpha)
                correction_threshold = getattr(p, "pag_correction_threshold", correction_threshold)
                correction_strength = getattr(p, "pag_correction_strength", correction_strength)
                tokens = getattr(p, "pag_tokens", tokens)
                p.extra_generation_params.update({
                        "PAG Active": active,
                        #"PAG AttnReg": attnreg,
                        #"PAG Tokens": tokens,
                        "PAG window_size Period": window_size,
                        "PAG CbS Score Threshold": correction_threshold,
                        "PAG CbS Correction Strength": correction_strength,
                        "PAG CTNMS Alpha": ctnms_alpha,
                        "PAG Step End": step_end,
                        "PAG EMA Smoothing Factor": ema_factor,
                })

                self.create_hook(p, active, attnreg, window_size, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end, p.width, p.height)

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

        def create_hook(self, p, active, attnreg, window_size, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end, width, height, *args, **kwargs):
                # Create a list of parameters for each concept
                pag_params = []

                #for _, strength in concept_conds:
                params = PAGStateParams()
                params.attnreg = attnreg 
                params.ema_smoothing_factor = ema_factor 
                params.step_end = step_end 
                params.window_size_period = window_size
                params.ctnms_alpha = ctnms_alpha
                params.correction_threshold = correction_threshold
                params.correction_strength = correction_strength
                params.strength = 1.0
                params.width = width
                params.height = height 
                params.dims = [width, height]
                pag_params.append(params)

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, pag_params)
                un = lambda params: self.unhook_callbacks()

                # Hook callbacks
                if ctnms_alpha > 0:
                        self.ready_hijack_forward(ctnms_alpha, width, height, ema_factor, step_end)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, *args, **kwargs):
                self.pag_postprocess_batch(p, *args, **kwargs)

        def pag_postprocess_batch(self, p, active, *args, **kwargs):
                self.unhook_callbacks()
                active = getattr(p, "pag_active", active)
                if active is False:
                        return

        def unhook_callbacks(self):
                global handles
                logger.debug('Unhooked callbacks')
                cross_attn_modules = self.get_cross_attn_modules()
                for module in cross_attn_modules:
                        self.remove_field_cross_attn_modules(module, 'pag_last_attn_map')
                        self.remove_field_cross_attn_modules(module, 'pag_step')
                        self.remove_field_cross_attn_modules(module, 'pag_step_end')
                        self.remove_field_cross_attn_modules(module, 'pag_ema_factor')
                        self.remove_field_cross_attn_modules(module, 'pag_ema')
                        _remove_all_forward_hooks(module, 'cross_token_non_maximum_suppression')
                script_callbacks.remove_current_script_callbacks()

        def apply_attnreg(self, f, C, alpha, B, *args, **kwargs):
                """
                Apply attention regulation on an embedding.

                Args:
                f (Tensor): The embedding tensor of shape (n, d).
                C (list): Indices of selected tokens.
                alpha (float): Attnreg strength.
                B (float): Lagrange multiplier B > 0
                gamma (int): Window size for the windowing function.

                Returns:
                Tensor: The corrected embedding tensor.
                """

                n, d = f.shape
                f_tilde = f.detach().clone()  # Copy the embedding tensor

                for token_idx, c in enumerate(C):
                        pass
                return f_tilde

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
                        Sc_flat_positive = Sc[Sc > 0] # product = greater positive value indicates more similarity, filter out values under score threshold from 0 to max

                        # calculate score threshold to filter out values under score threshold
                        # often there is a huge difference between the max and min values, so we use a log-like function instead
                        k = 10
                        e= 2.718281
                        pct_max = 1/(1+1e-10)
                        pct_min = 1e-16
                        # max of 0.999... to 0.0000...1
                        pct = min(pct_max, max(pct_min, 1 - e**(-k * percentile)))
                        tau = torch.quantile(Sc_flat_positive, pct)

                        Sc_tilde = Sc * (Sc > tau)  # Apply threshold and filter
                        Sc_tilde /= Sc_tilde.max()  # Normalize

                        window = psi(c, gamma, n, Sc_tilde.dtype, Sc_tilde.device).unsqueeze(1)  # Apply windowing function
                        Sc_tilde *= window
                        f_c_tilde = torch.sum(Sc_tilde * f, dim=0)  # Combine embeddings
                        f_tilde[c] = (1 - alpha) * f[c] + alpha * f_c_tilde  # Blend embeddings
                return f_tilde

        def ready_hijack_forward(self, alpha, width, height, ema_factor, step_end):
                """ Create a hook to modify the output of the forward pass of the cross attention module 
                Arguments:
                        alpha: float - The strength of the CTNMS correction, default 0.1
                        width: int - The width of the final output image map
                        height: int - The height of the final output image map
                        ema_factor: float - EMA smoothing factor, default 2.0
                        step_end: int - The number of steps to apply the CTNMS correction, after which don't

                Only modifies the output of the cross attention modules that get context (i.e. text embedding)
                """
                cross_attn_modules = self.get_cross_attn_modules()
                # add field for last_attn_map
                for module in cross_attn_modules:
                        self.add_field_cross_attn_modules(module, 'pag_last_attn_map', None)
                        self.add_field_cross_attn_modules(module, 'pag_step', torch.tensor([0]).to(device=shared.device))
                        self.add_field_cross_attn_modules(module, 'pag_step_end', torch.tensor([step_end]).to(device=shared.device))
                        self.add_field_cross_attn_modules(module, 'pag_ema', None)
                        self.add_field_cross_attn_modules(module, 'pag_ema_factor', torch.tensor([ema_factor]).to(device=shared.device, dtype=torch.float16))

                def cross_token_non_maximum_suppression(module, input, kwargs, output):
                        context = kwargs.get('context', None)
                        if context is None:
                                return

                        current_step = module.pag_step
                        end_step = module.pag_step_end
                        if current_step > end_step and end_step > 0:
                                return

                        batch_size, sequence_length, inner_dim = output.shape

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

                        if module.pag_ema is None:
                                module.pag_ema = output.detach().clone()

                        # Select token indices (Assuming this is provided as pag_params or similar)
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

                        # Calculate the EMA of the suppressed attention map
                        if module.pag_ema_factor > 0:
                                ema = module.pag_ema
                                ema_factor = module.pag_ema_factor / (1 + current_step)
                                # Add the suppressed attention map to the EMA
                                ema = ema_factor * ema + (1 - ema_factor) * suppressed_attention_map
                                module.pag_ema = ema
                                out_tensor = (1 -alpha) * output + (alpha) * ema
                                #out_tensor = (1-alpha) * ema + alpha * suppressed_attention_map
                        else:
                                out_tensor = (1-alpha) * output + alpha * suppressed_attention_map


                        # increment step
                        module.pag_step += 1

                        return out_tensor
                # Hook
                for module in cross_attn_modules:
                        handle = module.register_forward_hook(cross_token_non_maximum_suppression, with_kwargs=True)

        def get_cross_attn_modules(self):
                """ Get all cross attention modules """
                m = shared.sd_model
                nlm = m.network_layer_mapping
                cross_attn_modules = [m for m in nlm.values() if 'CrossAttention' in m.__class__.__name__]
                return cross_attn_modules

        def add_field_cross_attn_modules(self, module, field, value):
                """ Add a field to a module if it doesn't exist """
                if not hasattr(module, field):
                        setattr(module, field, value)
        
        def remove_field_cross_attn_modules(self, module, field):
                """ Remove a field from a module if it exists """
                if hasattr(module, field):
                        delattr(module, field)

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: list[PAGStateParams]):
                if isinstance(params.text_cond, dict):
                        text_cond = params.text_cond['crossattn'] # SD XL
                else:
                        text_cond = params.text_cond # SD 1.5

                sp = pag_params[0]
                window_size = sp.window_size_period
                correction_strength = sp.correction_strength
                score_threshold = sp.correction_threshold
                width = sp.width
                height = sp.height
                ctnms_alpha = sp.ctnms_alpha

                step = params.sampling_step
                step_end = sp.step_end

                if step > step_end:
                        return

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
                        xyz_grid.AxisOption("[PAG] Active", str, pag_apply_override('pag_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] ctnms_alpha", float, pag_apply_field("pag_ctnms_alpha")),
                        xyz_grid.AxisOption("[PAG] Step End", float, pag_apply_field("pag_step_end")),
                        xyz_grid.AxisOption("[PAG] Window Size", int, pag_apply_field("pag_window_size")),
                        xyz_grid.AxisOption("[PAG] Correction Threshold", float, pag_apply_field("pag_correction_threshold")),
                        xyz_grid.AxisOption("[PAG] Correction Strength", float, pag_apply_field("pag_correction_strength")),
                        xyz_grid.AxisOption("[PAG] CTNMS EMA Smoothing Factor", float, pag_apply_field("pag_ema_factor")),
                }
                return extra_axis_options

# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def pag_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def pag_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "pag_active"):
                setattr(p, "pag_active", True)
        setattr(p, field, x)
    return fun


# thanks torch; removing hooks DOESN'T WORK
# thank you to @ProGamerGov for this https://github.com/pytorch/pytorch/issues/70455
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