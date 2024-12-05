import logging
from os import environ
import math

import modules.scripts as scripts
import gradio as gr

from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams, AfterCFGCallbackParams
from modules.processing import StableDiffusionProcessing
from modules import shared

from scripts.ui_wrapper import UIWrapper
from scripts.incant_utils import module_hooks

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

incantations_debug = environ.get("INCANTAIONS_DEBUG", False)

"""
An unofficial implementation of "Eliminating Oversaturation and Artifacts of
High Guidance Scales in Diffusion Models"

@misc{sadat2024eliminatingoversaturationartifactshigh,
      title={Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models}, 
      author={Seyedmorteza Sadat and Otmar Hilliges and Romann M. Weber},
      year={2024},
      eprint={2410.02416},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.02416}, 
}

Parts of the code are based off the authors' implementation in the paper

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


handles = []
global_scale = 1

class APGStateParams:
        def __init__(self):
                self.apg_active: bool = False      # APG guidance scale
                self.momentum_buffer: MomentumBuffer = None
                self.eta: float = 0.
                self.norm_threshold: float = 0.
                self.apg_scale: int = -1      # APG guidance scale
                self.apg_momentum: float = 1.0
                self.apg_blur_threshold: float = 15.0 # 2^13 ~= 8192
                self.apg_start_step: int = 0
                self.apg_end_step: int = 150 
                self.crossattn_modules = [] # callable lambda


class APGExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.paste_field_names = []
                self.infotext_fields = []
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "Smoothed Energy Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Adaptive Projected Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='apg_active', info="Recommended to keep CFG Scale fixed at 3.0, use Sigma to adjust.")
                        with gr.Row():
                                apg_momentum = gr.Slider(value = -0.75, minimum = -1.0, maximum = 1.0, step = 0.1, label="APG Momentum", elem_id = 'apg_momentum', info="")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'apg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'apg_end_step', info="")

                params = [active, apg_momentum, start_step, end_step]
                                
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='APG Active' in d)),
                        (apg_momentum, 'APG Momentum'),
                        (start_step, 'APG Start Step'),
                        (end_step, 'APG End Step'),
                ]
                for p in params:
                        p.do_not_save_to_config = True
                        self.paste_field_names.append(p.elem_id)

                return params

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.apg_process_batch(p, *args, **kwargs)

        def apg_process_batch(self, p: StableDiffusionProcessing, active, apg_momentum, start_step, end_step, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "apg_active", active)
                if active is False:
                        return
                apg_momentum = getattr(p, "apg_momentum", apg_momentum)
                start_step = getattr(p, "apg_start_step", start_step)
                end_step = getattr(p, "apg_end_step", end_step)

                if active:
                        p.extra_generation_params.update({
                                "APG Active": active,
                                "APG Momentum": apg_momentum,
                                "APG Start Step": start_step,
                                "APG End Step": end_step,
                        })
                self.create_hook(p, active, apg_momentum, start_step, end_step)

        def create_hook(self, p: StableDiffusionProcessing, active, apg_momentum, start_step, end_step, *args, **kwargs):
                # Create a list of parameters for each concept
                apg_params = APGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['apg_params'] = apg_params
                
                apg_params.apg_active = active 
                apg_params.apg_momentum = apg_momentum
                apg_params.apg_blur_threshold = 10.5
                apg_params.apg_start_step = start_step
                apg_params.apg_end_step = end_step

                apg_params.momentum_buffer = MomentumBuffer(apg_momentum) 
                apg_params.eta = p.eta
                apg_params.norm_threshold = 0.


                logger.debug('Hooked callbacks')

        def postprocess_batch(self, p, *args, **kwargs):
                self.apg_postprocess_batch(p, *args, **kwargs)

        def apg_postprocess_batch(self, p, active, apg_momentum, start_step, end_step, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()

                logger.debug('Removed script callbacks')
                active = getattr(p, "apg_active", active)
                if active is False:
                        return

        def remove_all_hooks(self):
                self_attn_modules = self.get_cross_attn_modules()
                for module in self_attn_modules:
                        module_hooks.modules_remove_field(module.to_q, 'apg_enable')
                        module_hooks.modules_remove_field(module.to_q, 'apg_parent_module')
                        module_hooks.remove_module_forward_hook(module.to_q, 'apg_to_q_hook')

        def unhook_callbacks(self, apg_params: APGStateParams):
                global handles
                return

        def ready_hijack_forward(self, selfattn_modules, apg_momentum, apg_blur_threshold, height, width):
                for module in selfattn_modules:
                        module_hooks.modules_add_field(module.to_q, 'apg_enable', False)
                        module_hooks.modules_add_field(module.to_q, 'apg_parent_module', [module])

                def apg_to_q_hook(module, input, kwargs, output):
                        if not hasattr(module, 'apg_enable'):
                                return
                        if not module.apg_enable:
                                return
                        batch_size, seq_len, inner_dim = input[0].shape
                        h = module.apg_parent_module[0].heads
                        head_dim = inner_dim // h

                        module_attn_size = seq_len
                        downscale_h = int((module_attn_size * (height / width)) ** 0.5)
                        downscale_w = module_attn_size // downscale_h

                        # actual sigma value is calculated as 2 ^ sigma
                        is_inf_blur = apg_momentum > apg_blur_threshold
                        momentum_exp = 2 ** apg_momentum
                        kernel_size = math.ceil(6 * momentum_exp) + 1 - math.ceil(6 * momentum_exp) % 2

                        q_uncond, q= output.chunk(2, dim=0) 
                        q = q.view(batch_size//2, -1, h, head_dim).transpose(1, 2) # (batch, num_heads, seq_len, head_dim)
                        q = q.permute(0, 1, 3, 2).reshape(batch_size//2 * h, head_dim, downscale_h, downscale_w) # (batch * num_heads, head_dim, height, width)

                        if is_inf_blur:
                                q = gaussian_blur_inf(q, 1.0, momentum_exp)
                        else:
                                q = gaussian_blur_2d(q, kernel_size, momentum_exp)

                        q = q.reshape(batch_size // 2, h, head_dim, downscale_h * downscale_w) # (batch, num_heads, head_dim, seq_len)
                        q = q.view(batch_size // 2, h * head_dim, seq_len).transpose(1, 2) # (batch, inner_dim, seq_len)
                        q = torch.cat((q_uncond, q), dim=0)

                        return q

                # Create hooks 
                for module in selfattn_modules:
                        module_hooks.module_add_forward_hook(module.to_q, apg_to_q_hook, hook_type="forward", with_kwargs=True)

        def get_middle_block_modules(self):
                """ Get all attention modules from the middle block 
                Refere to page 22 of the APG paper, Appendix A.2
                
                """
                middle_block_modules = module_hooks.get_modules(
                        network_layer_name_filter = 'middle_block_',
                        module_name_filter = 'CrossAttention'
                )
                middle_block_modules = [m for m in middle_block_modules if 'attn1' in m.network_layer_name]
                return middle_block_modules

        def get_cross_attn_modules(self):
                """ Get all cross attention modules """
                return self.get_middle_block_modules()

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, apg_params: APGStateParams):
                # always unhook
                self.unhook_callbacks(apg_params)
                if not apg_params.apg_active:
                        return

                in_interval = apg_params.apg_start_step <= params.sampling_step <= apg_params.apg_end_step
                for module in apg_params.crossattn_modules:
                        if hasattr(module.to_q, 'apg_enable'):
                                module.to_q.apg_enable = in_interval

        def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams, apg_params: APGStateParams):
                pass

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[APG] Active", str, apg_apply_override('apg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[APG] APG Momentum", float, apg_apply_field("apg_momentum")),
                        xyz_grid.AxisOption("[APG] APG Start Step", int, apg_apply_field("apg_start_step")),
                        xyz_grid.AxisOption("[APG] APG End Step", int, apg_apply_field("apg_end_step")),
                }
                return extra_axis_options


# from modules/sd_samplers_cfg_denoiser.py:187-195
def get_make_condition_dict_fn(text_uncond):
        if shared.sd_model.model.conditioning_key == "crossattn-adm":
                make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": [c_crossattn], "c_adm": c_adm}
        else:
                if isinstance(text_uncond, dict):
                        make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
                else:
                        make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}
        return make_condition_dict


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def apg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "apg_active"):
                setattr(p, "apg_active", True)
        if 'cfg_interval_' in field and not hasattr(p, "cfg_interval_enable"):
            setattr(p, "cfg_interval_enable", True)
    return fun


def apg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "apg_active"):
                setattr(p, "apg_active", True)
        setattr(p, field, x)
    return fun


# Gaussian blur
# taken from https://github.com/SusungHong/APG-SDXL/blob/master/pipeline_seg.py
def gaussian_blur_2d(img, kernel_size, sigma):
        height = img.shape[-1]
        kernel_size = min(kernel_size, height - (height % 2 - 1))
        ksize_half = (kernel_size - 1) * 0.5

        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

        pdf = torch.exp(-0.5 * (x / sigma).pow(2))

        x_kernel = pdf / pdf.sum()
        x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

        kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
        kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

        padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

        img = F.pad(img, padding, mode="reflect")
        img = F.conv2d(img, kernel2d, groups=img.shape[-3])

        return img


def gaussian_blur_inf(img, kernel_size, sigma):
        img[:] = img.mean(dim=(-2, -1), keepdim=True)

        return img

# taken directly from the paper
class MomentumBuffer:
        def __init__(self, momentum: float):
                self.momentum = momentum
                self.running_average = None
        def update(self, update_value: torch.Tensor):
                if self.running_average is None:
                        self.running_average = torch.zeros_like(update_value)
                new_average = self.momentum * self.running_average
                self.running_average = update_value + new_average


# taken directly from the paper
def project(
        v0: torch.Tensor, # [B, C, H, W]
        v1: torch.Tensor, # [B, C, H, W]
        ):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


# modifed from the paper
def normalized_guidance(
        pred_cond: torch.Tensor, # [B, C, H, W]
        pred_uncond: torch.Tensor, # [B, C, H, W]
        diff: torch.Tensor, # [B, C, H, W],
        guidance_scale: float,
        momentum_buffer: MomentumBuffer = None,
        eta: float = 1.0,
        norm_threshold: float = 0.0,
        ):
        pred_cond = pred_cond.unsqueeze(0)
        pred_uncond = pred_uncond.unsqueeze(0)
        diff = diff.unsqueeze(0)
        eta = eta or 1.0
        # diff = pred_cond - pred_uncond
        if momentum_buffer is not None:
                momentum_buffer.update(diff)
                diff = momentum_buffer.running_average
        if norm_threshold > 0:
                ones = torch.ones_like(diff)
                diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
                diff = diff * scale_factor
        diff_parallel, diff_orthogonal = project(diff, pred_cond)
        normalized_update = diff_orthogonal + eta * diff_parallel
        #return normalized_update.squeeze(0)
        pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
        #pred_cond = pred_cond.squeeze(0)
        #pred_uncond = pred_uncond.squeeze(0)
        return pred_guided.squeeze(0)

