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
                self.apg_parallel_scale: float = 1.0      # APG scale for paralllel guidance
                self.apg_momentum: float = 1.0
                self.apg_start_step: int = 0
                self.apg_end_step: int = 150 


class APGExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.paste_field_names = []
                self.infotext_fields = []
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "Adaptive Projected Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Adaptive Projected Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='apg_active', info="")
                        with gr.Row():
                                apg_momentum = gr.Slider(value = -0.5, minimum = -1.5, maximum = 1.5, step = 0.05, label="APG Momentum", elem_id = 'apg_momentum', info="Recommended between [-0.75, -0.25]")
                                apg_norm_threshold = gr.Slider(value = 15.0, minimum = 0.0, maximum = 20.0, step = 0.05, label="APG Norm Threshold", elem_id = 'apg_norm_threshold', info="Rescaling factor, recommended values between (0.25, 10), 0 is \u221E")
                                apg_parallel_scale = gr.Slider(value = 0.0, minimum = 0.0, maximum = 1.0, step = 0.05, label="APG Parallel Scale", elem_id = 'apg_parallel_scale', info="Scale of parallel CFG, 1.0 is equivalent to CFG")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'apg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'apg_end_step', info="")

                params = [active, apg_momentum, apg_norm_threshold, apg_parallel_scale, start_step, end_step]
                                
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='APG Active' in d)),
                        (apg_momentum, 'APG Momentum'),
                        (apg_norm_threshold, 'APG Norm Threshold'),
                        (apg_parallel_scale, 'APG Parallel Scale'),
                        (start_step, 'APG Start Step'),
                        (end_step, 'APG End Step'),
                ]
                for p in params:
                        p.do_not_save_to_config = True
                        self.paste_field_names.append(p.elem_id)

                return params

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.apg_process_batch(p, *args, **kwargs)

        def apg_process_batch(self, p: StableDiffusionProcessing, active, apg_momentum, apg_norm_threshold, apg_parallel_scale, start_step, end_step, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "apg_active", active)
                if active is False:
                        return
                apg_momentum = getattr(p, "apg_momentum", apg_momentum)
                apg_norm_threshold = getattr(p, "apg_norm_threshold", apg_norm_threshold)
                apg_parallel_scale = getattr(p, "apg_parallel_scale", apg_parallel_scale)
                start_step = getattr(p, "apg_start_step", start_step)
                end_step = getattr(p, "apg_end_step", end_step)

                if active:
                        p.extra_generation_params.update({
                                "APG Active": active,
                                "APG Momentum": apg_momentum,
                                "APG Norm Threshold": apg_norm_threshold,
                                "APG Start Step": start_step,
                                "APG End Step": end_step,
                        })
                self.create_hook(p, active, apg_momentum, apg_norm_threshold, apg_parallel_scale, start_step, end_step)

        def create_hook(self, p: StableDiffusionProcessing, active, apg_momentum, apg_norm_threshold,apg_parallel_scale, start_step, end_step, *args, **kwargs):
                # Create a list of parameters for each concept
                apg_params = APGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['apg_params'] = apg_params
                
                apg_params.apg_active = active 
                apg_params.apg_momentum = apg_momentum
                apg_params.norm_threshold = apg_norm_threshold
                apg_params.apg_parallel_scale = apg_parallel_scale
                apg_params.apg_blur_threshold = 10.5
                apg_params.apg_start_step = start_step
                apg_params.apg_end_step = end_step

                apg_params.momentum_buffer = MomentumBuffer(apg_momentum) 
                apg_params.eta = p.eta


                logger.debug('Hooked callbacks')

        def postprocess_batch(self, p, *args, **kwargs):
                self.apg_postprocess_batch(p, *args, **kwargs)

        def apg_postprocess_batch(self, p, active, apg_momentum, apg_norm_threshold,apg_parallel_scale, start_step, end_step, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()

                logger.debug('Removed script callbacks')
                active = getattr(p, "apg_active", active)
                if active is False:
                        return

        def remove_all_hooks(self):
                return

        def unhook_callbacks(self, apg_params: APGStateParams):
                global handles
                return

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[APG] Active", str, apg_apply_override('apg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[APG] APG Momentum", float, apg_apply_field("apg_momentum")),
                        xyz_grid.AxisOption("[APG] APG Norm Threshold", float, apg_apply_field("apg_norm_threshold")),
                        xyz_grid.AxisOption("[APG] APG Parallel Scale", float, apg_apply_field("apg_parallel_scale")),
                        xyz_grid.AxisOption("[APG] APG Start Step", int, apg_apply_field("apg_start_step")),
                        xyz_grid.AxisOption("[APG] APG End Step", int, apg_apply_field("apg_end_step")),
                }
                return extra_axis_options


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
        #diff: torch.Tensor, # [B, C, H, W],
        guidance_scale: float,
        momentum_buffer: MomentumBuffer = None,
        eta: float = 1.0,
        norm_threshold: float = 0.0,
        ):
        pred_cond = pred_cond.unsqueeze(0)
        pred_uncond = pred_uncond.unsqueeze(0)
        #diff = diff.unsqueeze(0)
        diff = pred_cond - pred_uncond
        if momentum_buffer is not None:
                momentum_buffer.update(diff)
                diff = momentum_buffer.running_average
        if norm_threshold > 0:
                ones = torch.ones_like(diff)
                diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
                scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
                diff = diff * scale_factor
        diff_parallel, diff_orthogonal = project(diff, pred_uncond)
        normalized_update = diff_orthogonal + eta * diff_parallel
        return normalized_update.squeeze(0)
        #pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
        #pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
        #pred_cond = pred_cond.squeeze(0)
        #pred_uncond = pred_uncond.squeeze(0)
        return pred_guided.squeeze(0)

