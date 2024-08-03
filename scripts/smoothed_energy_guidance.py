import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import scipy.stats as stats

from scripts.ui_wrapper import UIWrapper, arg
from modules import script_callbacks, patches
from modules.hypernetworks import hypernetwork
#import modules.sd_hijack_optimizations
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
from modules.prompt_parser import reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import catenate_conds
from modules.sd_samplers_cfg_denoiser import CFGDenoiser
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

incantations_debug = environ.get("INCANTAIONS_DEBUG", False)

"""
An unofficial implementation of "Smoothed Energy Guidance for SDXL" for Automatic1111 WebUI.

@article{hong2024smoothed,
  title={Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention},
  author={Hong, Susung},
  journal={arXiv preprint arXiv:2408.00760},
  year={2024}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


handles = []
global_scale = 1

class SEGStateParams:
        def __init__(self):
                self.seg_active: bool = False      # SEG guidance scale
                self.seg_sanf: bool = False # saliency-adaptive noise fusion, handled in cfg_combiner
                self.seg_scale: int = -1      # SEG guidance scale
                self.seg_blur_sigma: float = 1.0
                self.seg_blur_threshold: float = 15.0 # 2^13 ~= 8192
                self.seg_start_step: int = 0
                self.seg_end_step: int = 150 
                self.step : int = 0 
                self.max_sampling_step : int = 1 
                self.guidance_scale: int = -1 # CFG
                self.current_noise_level: float = 100.0
                self.x_in = None
                self.text_cond = None
                self.image_cond = None
                self.sigma = None
                self.text_uncond = None
                self.make_condition_dict = None # callable lambda
                self.crossattn_modules = [] # callable lambda
                self.to_v_modules = []
                self.to_out_modules = []
                self.seg_x_out = None
                self.batch_size = -1      # Batch size
                self.denoiser = None # CFGDenoiser
                self.patched_combine_denoised = None
                self.conds_list = None
                self.uncond_shape_0 = None


class SEGExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "Smoothed Energy Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Smoothed Energy Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='seg_active')
                        seg_sanf = gr.Checkbox(value=False, default=False, label="Use Saliency-Adaptive Noise Fusion", elem_id='seg_sanf')
                        with gr.Row():
                                seg_scale = gr.Slider(value = 0, minimum = 0, maximum = 20.0, step = 0.5, label="SEG Scale", elem_id = 'seg_scale', info="")
                        with gr.Row():
                                seg_blur_sigma = gr.Slider(value = 1.0, minimum = 0.0, maximum = 6.0, step = 0.5, label="SEG Blur Sigma", elem_id = 'seg_blur_sigma', info="")
                                seg_blur_threshold = gr.Slider(value = 14.0, minimum = 0, maximum = 14.0, step = 0.5, label="SEG Blur Threshold", elem_id = 'seg_blur_threshold', info="Values >= 14 are infinite blur")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'seg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'seg_end_step', info="")

                params = [active, seg_sanf, seg_scale, seg_blur_sigma, seg_blur_threshold, start_step, end_step]
                                
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='SEG Active' in d)),
                        (seg_sanf, lambda d: gr.Checkbox.update(value='SEG SANF' in d)),
                        (seg_scale, 'SEG Scale'),
                        (seg_blur_sigma, 'SEG Blur Sigma'),
                        (seg_blur_threshold, 'SEG Blur Threshold'),
                        (start_step, 'SEG Start Step'),
                        (end_step, 'SEG End Step'),
                ]
                for p in params:
                        p.do_not_save_to_config = True
                        self.paste_field_names.append(p.elem_id)

                return params

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.seg_process_batch(p, *args, **kwargs)

        def seg_process_batch(self, p: StableDiffusionProcessing, active, seg_scale, seg_blur_sigma, seg_blur_threshold, start_step, end_step, seg_sanf, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "seg_active", active)
                seg_sanf = getattr(p, "seg_sanf", seg_sanf)
                if active is False:
                        return
                seg_scale = getattr(p, "seg_scale", seg_scale)
                seg_blur_sigma = getattr(p, "seg_blur_sigma", seg_blur_sigma)
                seg_blur_threshold = getattr(p, "seg_blur_threshold", seg_blur_threshold)
                start_step = getattr(p, "seg_start_step", start_step)
                end_step = getattr(p, "seg_end_step", end_step)

                if active:
                        p.extra_generation_params.update({
                                "SEG Active": active,
                                "SEG SANF": seg_sanf,
                                "SEG Scale": seg_scale,
                                "SEG Blur Sigma": seg_blur_sigma,
                                "SEG Blur Threshold": seg_blur_threshold,
                                "SEG Start Step": start_step,
                                "SEG End Step": end_step,
                        })
                self.create_hook(p, active, seg_scale, seg_blur_sigma, seg_blur_threshold, start_step, end_step, seg_sanf)

        def create_hook(self, p: StableDiffusionProcessing, active, seg_scale, seg_blur_sigma, seg_blur_threshold, start_step, end_step, seg_sanf, *args, **kwargs):
                # Create a list of parameters for each concept
                seg_params = SEGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['seg_params'] = seg_params
                
                seg_params.seg_active = active 
                seg_params.seg_sanf = seg_sanf 
                seg_params.seg_scale = seg_scale
                seg_params.seg_blur_sigma = seg_blur_sigma
                seg_params.seg_blur_threshold = seg_blur_threshold
                seg_params.seg_start_step = start_step
                seg_params.seg_end_step = end_step

                seg_params.max_sampling_step = p.steps
                seg_params.guidance_scale = p.cfg_scale
                seg_params.batch_size = p.batch_size
                seg_params.denoiser = None
                seg_params.cfg_interval_scheduled_value = p.cfg_scale

                # Get all the qv modules
                cross_attn_modules = self.get_cross_attn_modules()
                if len(cross_attn_modules) == 0:
                        logger.error("No cross attention modules found, cannot proceed with SEG")
                        return
                seg_params.crossattn_modules = [m for m in cross_attn_modules if 'CrossAttention' in m.__class__.__name__]

                # Use lambda to call the callback function with the parameters to avoid global variables
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, seg_params)
                cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(callback_params, seg_params)
                #after_cfg_lambda = lambda x: self.cfg_after_cfg_callback(x, params)
                unhook_lambda = lambda _: self.unhook_callbacks(seg_params)

                if seg_params.seg_active:
                        self.ready_hijack_forward(seg_params.crossattn_modules, seg_scale)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_cfg_denoised(cfg_denoised_lambda)
                #script_callbacks.on_cfg_after_cfg(after_cfg_lambda)
                script_callbacks.on_script_unloaded(unhook_lambda)

        def postprocess_batch(self, p, *args, **kwargs):
                self.seg_postprocess_batch(p, *args, **kwargs)

        def seg_postprocess_batch(self, p, active, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()

                logger.debug('Removed script callbacks')
                active = getattr(p, "seg_active", active)
                if active is False:
                        return

        def remove_all_hooks(self):
                cross_attn_modules = self.get_cross_attn_modules()
                for module in cross_attn_modules:
                        to_v = getattr(module, 'to_v', None)
                        self.remove_field_cross_attn_modules(module, 'seg_enable')
                        self.remove_field_cross_attn_modules(module, 'seg_last_to_v')
                        self.remove_field_cross_attn_modules(to_v, 'seg_parent_module')
                        _remove_all_forward_hooks(module, 'seg_pre_hook')
                        _remove_all_forward_hooks(to_v, 'to_v_pre_hook')

        def unhook_callbacks(self, seg_params: SEGStateParams):
                global handles
                return


        def ready_hijack_forward(self, crossattn_modules, seg_scale):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                Then applies the SEG perturbation to the output of the cross attention module (multiplication by identity)
                """

                # add field for last_to_v
                for module in crossattn_modules:
                        to_v = getattr(module, 'to_v', None)
                        self.add_field_cross_attn_modules(module, 'seg_enable', False)
                        self.add_field_cross_attn_modules(module, 'seg_last_to_v', None)
                        self.add_field_cross_attn_modules(to_v, 'seg_parent_module', [module])
                        # self.add_field_cross_attn_modules(to_out, 'seg_parent_module', [module])

                def to_v_pre_hook(module, input, kwargs, output):
                        """ Copy the output of the to_v module to the parent module """
                        parent_module = getattr(module, 'seg_parent_module', None)
                        # copy the output of the to_v module to the parent module
                        setattr(parent_module[0], 'seg_last_to_v', output.detach().clone())

                def seg_pre_hook(module, input, kwargs, output):
                        if hasattr(module, 'seg_enable') and getattr(module, 'seg_enable', False) is False:
                                return
                        if not hasattr(module, 'seg_last_to_v'):
                                # oops we forgot to unhook
                                return

                        # get the last to_v output and save it
                        last_to_v = getattr(module, 'seg_last_to_v', None)

                        batch_size, seq_len, inner_dim = output.shape
                        identity = torch.eye(seq_len, dtype=last_to_v.dtype, device=shared.device).expand(batch_size, -1, -1)
                        if last_to_v is not None:    
                                new_output = torch.einsum('bij,bjk->bik', identity, last_to_v[:, :seq_len, :])
                                return new_output
                        else:
                                # this is bad
                                return output

                # Create hooks 
                for module in crossattn_modules:
                        handle_parent = module.register_forward_hook(seg_pre_hook, with_kwargs=True)
                        to_v = getattr(module, 'to_v', None)
                        handle_to_v = to_v.register_forward_hook(to_v_pre_hook, with_kwargs=True)

        def get_middle_block_modules(self):
                """ Get all attention modules from the middle block 
                Refere to page 22 of the SEG paper, Appendix A.2
                
                """
                try:
                        m = shared.sd_model
                        nlm = m.network_layer_mapping
                        middle_block_modules = [m for m in nlm.values() if 'middle_block_1_transformer_blocks_0_attn1' in m.network_layer_name and 'CrossAttention' in m.__class__.__name__]
                        return middle_block_modules
                except AttributeError:
                        logger.exception("AttributeError in get_middle_block_modules", stack_info=True)
                        return []
                except Exception:
                        logger.exception("Exception in get_middle_block_modules", stack_info=True)
                        return []

        def get_cross_attn_modules(self):
                """ Get all cross attention modules """
                return self.get_middle_block_modules()

        def add_field_cross_attn_modules(self, module, field, value):
                """ Add a field to a module if it doesn't exist """
                if not hasattr(module, field):
                        setattr(module, field, value)
        
        def remove_field_cross_attn_modules(self, module, field):
                """ Remove a field from a module if it exists """
                if hasattr(module, field):
                        delattr(module, field)

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, seg_params: SEGStateParams):
                # always unhook
                self.unhook_callbacks(seg_params)

                seg_params.step = params.sampling_step

                # Run SEG only if active and within interval
                if not seg_params.seg_active or seg_params.seg_scale <= 0:
                        return
                if not seg_params.seg_start_step <= params.sampling_step <= seg_params.seg_end_step or seg_params.seg_scale <= 0:
                        return

                if isinstance(params.text_cond, dict):
                        text_cond = params.text_cond['crossattn'] # SD XL
                        seg_params.text_cond = {}
                        seg_params.text_uncond = {}
                        for key, value in params.text_cond.items():
                                seg_params.text_cond[key] = value.clone().detach()
                                seg_params.text_uncond[key] = value.clone().detach()
                else:
                        text_cond = params.text_cond # SD 1.5
                        seg_params.text_cond = text_cond.clone().detach()
                        seg_params.text_uncond = text_cond.clone().detach()

                seg_params.x_in = params.x.clone().detach()
                seg_params.sigma = params.sigma.clone().detach()
                seg_params.image_cond = params.image_cond.clone().detach()
                seg_params.denoiser = params.denoiser
                seg_params.make_condition_dict = get_make_condition_dict_fn(params.text_uncond)


        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, seg_params: SEGStateParams):
                """ Callback function for the CFGDenoisedParams 
                Refer to pg.22 A.2 of the SEG paper for how CFG and SEG combine
                
                """
                # Run only within interval
                # Run SEG only if active and within interval
                if not seg_params.seg_active or seg_params.seg_scale <= 0:
                        return
                if not seg_params.seg_start_step <= params.sampling_step <= seg_params.seg_end_step or seg_params.seg_scale <= 0:
                        return

                # passed from on_cfg_denoiser_callback
                x_in = seg_params.x_in
                tensor = seg_params.text_cond
                uncond = seg_params.text_uncond
                image_cond_in = seg_params.image_cond
                sigma_in = seg_params.sigma
                
                # concatenate the conditions 
                # "modules/sd_samplers_cfg_denoiser.py:237"
                cond_in = catenate_conds([tensor, uncond])
                make_condition_dict = get_make_condition_dict_fn(uncond)
                conds = make_condition_dict(cond_in, image_cond_in)
                
                # set seg_enable to True for the hooked cross attention modules
                for module in seg_params.crossattn_modules:
                        setattr(module, 'seg_enable', True)

                # get the SEG guidance (is there a way to optimize this so we don't have to calculate it twice?)
                seg_x_out = params.inner_model(x_in, sigma_in, cond=conds)

                # update seg_x_out
                seg_params.seg_x_out = seg_x_out

                # set seg_enable to False
                for module in seg_params.crossattn_modules:
                        setattr(module, 'seg_enable', False)
        
        def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams, seg_params: SEGStateParams):
                #self.unhook_callbacks(seg_params)
                pass

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[SEG] Active", str, seg_apply_override('seg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SEG] SANF", str, seg_apply_override('seg_sanf', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SEG] SEG Scale", float, seg_apply_field("seg_scale")),
                        xyz_grid.AxisOption("[SEG] SEG Start Step", int, seg_apply_field("seg_start_step")),
                        xyz_grid.AxisOption("[SEG] SEG End Step", int, seg_apply_field("seg_end_step")),
                }
                return extra_axis_options


def combine_denoised_pass_conds_list(*args, **kwargs):
        """ Hijacked function for combine_denoised in CFGDenoiser """
        original_func = kwargs.get('original_func', None)
        new_params = kwargs.get('seg_params', None)

        if new_params is None:
                logger.error("new_params is None")
                return original_func(*args)

        def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
                denoised_uncond = x_out[-uncond.shape[0]:]
                denoised = torch.clone(denoised_uncond)

                noise_level = calculate_noise_level(new_params.step, new_params.max_sampling_step)

                # Calculate CFG Scale
                cfg_scale = cond_scale
                new_params.cfg_interval_scheduled_value = cfg_scale

                if new_params.cfg_interval_enable:
                        if new_params.cfg_interval_schedule != 'Constant':
                                # Calculate noise interval
                                start = new_params.cfg_interval_low
                                end = new_params.cfg_interval_high
                                begin_range = start if start <= end else end
                                end_range = end if start <= end else start
                                # Scheduled CFG Value
                                scheduled_cfg_scale = cfg_scheduler(new_params.cfg_interval_schedule, new_params.step, new_params.max_sampling_step, cond_scale)
                                # Only apply CFG in the interval
                                cfg_scale = scheduled_cfg_scale if begin_range <= noise_level <= end_range else 1.0
                                new_params.cfg_interval_scheduled_value = scheduled_cfg_scale

                # This may be temporarily necessary for compatibility with scfg
                # if not new_params.seg_start_step <= new_params.step <= new_params.seg_end_step:
                #        return original_func(*args)

                # This may be temporarily necessary for compatibility with scfg
                # if not new_params.seg_start_step <= new_params.step <= new_params.seg_end_step:
                #        return original_func(*args)

                if incantations_debug:
                        logger.debug(f"Schedule: {new_params.cfg_interval_schedule}, CFG Scale: {cfg_scale}, Noise_level: {round(noise_level,3)}")

                for i, conds in enumerate(conds_list):
                        for cond_index, weight in conds:
                                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cfg_scale)

                                # Apply SEG guidance only within interval
                                if not new_params.seg_start_step <= new_params.step <= new_params.seg_end_step or new_params.seg_scale <= 0:
                                        continue
                                else:
                                        try:
                                                denoised[i] += (x_out[cond_index] - new_params.seg_x_out[i]) * (weight * new_params.seg_scale)
                                        except TypeError:
                                                logger.exception("TypeError in combine_denoised_pass_conds_list")
                                        except IndexError:
                                                logger.exception("IndexError in combine_denoised_pass_conds_list")
                                        #logger.debug(f"added SEG guidance to denoised - seg_scale:{global_scale}")
                return denoised
        return new_combine_denoised(*args)


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
def seg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "seg_active"):
                setattr(p, "seg_active", True)
        if 'cfg_interval_' in field and not hasattr(p, "cfg_interval_enable"):
            setattr(p, "cfg_interval_enable", True)
    return fun


def seg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "seg_active"):
                setattr(p, "seg_active", True)
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
