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

from scripts.incant_utils import module_hooks

# from pytorch_memlab import LineProfiler, MemReporter
# reporter = MemReporter()

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

incantations_debug = environ.get("INCANTAIONS_DEBUG", False)


"""
An unofficial implementation of "Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidancee" for Automatic1111 WebUI.

This builds upon the code provided in the official S-CFG repository: https://github.com/SmilesDZgk/S-CFG


@inproceedings{shen2024rethinking,
  title={Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidancee},
  author={Shen, Dazhong and Song, Guanglu and Xue, Zeyue and Wang, Fu-Yun and Liu, Yu},
  booktitle={Proceedings of The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024}
}

Parts of the code are based on Diffusers under the Apache License 2.0:
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


handles = []
global_scale = 1

SCFG_MODULES = ['to_q', 'to_k']


class SCFGStateParams:
        def __init__(self):
                self.scfg_scale:float = 0.8
                self.rate_min = 0.8
                self.rate_max = 3.0
                self.rate_clamp = 15.0
                self.R = 4
                self.start_step = 0
                self.end_step = 150 

                self.max_sampling_steps = -1
                self.current_step = 0
                self.height = -1 
                self.width = -1 

                self.statistics = {
                        "min_rate": float('inf'), 
                        "max_rate": float('-inf'), 
                }

                self.mask_t = None
                self.mask_fore = None
                self.denoiser = None
                self.all_crossattn_modules = None
                self.patched_combined_denoised = None


class SCFGExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "S-CFG"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('S-CFG', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='scfg_active')
                        with gr.Row():
                                scfg_scale = gr.Slider(value = 1.0, minimum = 0, maximum = 10.0, step = 0.1, label="SCFG Scale", elem_id = 'scfg_scale', info="")
                                scfg_r = gr.Slider(value = 4, minimum = 1, maximum = 16, step = 1, label="SCFG R", elem_id = 'scfg_r', info="Scale factor. Greater R uses more memory.")
                        with gr.Row():
                                scfg_rate_min = gr.Slider(value = 0.8, minimum = 0, maximum = 30.0, step = 0.1, label="Min Rate", elem_id = 'scfg_rate_min', info="")
                                scfg_rate_max = gr.Slider(value = 3.0, minimum = 0, maximum = 30.0, step = 0.1, label="Max Rate", elem_id = 'scfg_rate_max', info="")
                                scfg_rate_clamp = gr.Slider(value = 0.0, minimum = 0, maximum = 30.0, step = 0.1, label="Clamp Rate", elem_id = 'scfg_rate_clamp', info="If > 0, clamp max rate to Clamp Rate / CFG Scale. Overrides max rate.")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'scfg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'scfg_end_step', info="")
                                
                active.do_not_save_to_config = True
                scfg_scale.do_not_save_to_config = True
                scfg_rate_min.do_not_save_to_config = True
                scfg_rate_max.do_not_save_to_config = True
                scfg_rate_clamp.do_not_save_to_config = True
                scfg_r.do_not_save_to_config = True
                start_step.do_not_save_to_config = True
                end_step.do_not_save_to_config = True

                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='SCFG Active' in d)),
                        (scfg_scale, 'SCFG Scale'),
                        (scfg_rate_min, 'SCFG Rate Min'),
                        (scfg_rate_max, 'SCFG Rate Max'),
                        (scfg_rate_clamp, 'SCFG Rate Clamp'),
                        (start_step, 'SCFG Start Step'),
                        (end_step, 'SCFG End Step'),
                        (scfg_r, 'SCFG R'),
                ]
                self.paste_field_names = [
                        'scfg_active',
                        'scfg_scale',
                        'scfg_rate_min',
                        'scfg_rate_max',
                        'scfg_rate_clamp',
                        'scfg_start_step',
                        'scfg_end_step',
                        'scfg_r',
                ]
                return [active, scfg_scale, scfg_rate_min, scfg_rate_max, scfg_rate_clamp, start_step, end_step, scfg_r]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.pag_process_batch(p, *args, **kwargs)

        def pag_process_batch(self, p: StableDiffusionProcessing, active, scfg_scale, scfg_rate_min, scfg_rate_max, scfg_rate_clamp, start_step, end_step, scfg_r, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "scfg_active", active)
                if active is False:
                        return
                scfg_scale = getattr(p, "scfg_scale", scfg_scale)
                scfg_rate_min = getattr(p, "scfg_rate_min", scfg_rate_min)
                scfg_rate_max = getattr(p, "scfg_rate_max", scfg_rate_max)
                scfg_rate_clamp = getattr(p, "scfg_rate_clamp", scfg_rate_clamp)
                start_step = getattr(p, "scfg_start_step", start_step)
                end_step = getattr(p, "scfg_end_step", end_step)
                scfg_r = getattr(p, "scfg_r", scfg_r)

                p.extra_generation_params.update({
                        "SCFG Active": active,
                        "SCFG Scale": scfg_scale,
                        "SCFG Rate Min": scfg_rate_min,
                        "SCFG Rate Max": scfg_rate_max,
                        "SCFG Rate Clamp": scfg_rate_clamp,
                        "SCFG Start Step": start_step,
                        "SCFG End Step": end_step,
                        "SCFG R": scfg_r,
                })
                self.create_hook(p, active, scfg_scale, scfg_rate_min, scfg_rate_max, scfg_rate_clamp, start_step, end_step, scfg_r)

        def create_hook(self, p: StableDiffusionProcessing, active, scfg_scale, scfg_rate_min, scfg_rate_max, scfg_rate_clamp, start_step, end_step, scfg_r):
                # Create a list of parameters for each concept
                scfg_params = SCFGStateParams()

                # Add to p
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['scfg_params'] = scfg_params

                scfg_params.denoiser = None
                scfg_params.all_crossattn_modules = self.get_all_crossattn_modules()
                scfg_params.max_sampling_steps = p.steps
                scfg_params.scfg_scale = scfg_scale
                scfg_params.rate_min = scfg_rate_min
                scfg_params.rate_max = scfg_rate_max
                scfg_params.rate_clamp = scfg_rate_clamp
                scfg_params.start_step = start_step
                scfg_params.end_step = end_step
                scfg_params.R = scfg_r
                scfg_params.height = p.height
                scfg_params.width = p.width

                # Use lambda to call the callback function with the parameters to avoid global variables
                #cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, scfg_params)
                cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(callback_params, scfg_params)
                unhook_lambda = lambda _: self.unhook_callbacks(scfg_params)

                self.ready_hijack_forward(scfg_params.all_crossattn_modules)

                logger.debug('Hooked callbacks')
                #script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_cfg_denoised(cfg_denoised_lambda)
                script_callbacks.on_script_unloaded(unhook_lambda)

        def postprocess_batch(self, p, *args, **kwargs):
                self.scfg_postprocess_batch(p, *args, **kwargs)

        def scfg_postprocess_batch(self, p, active, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()

                logger.debug('Removed script callbacks')
                active = getattr(p, "scfg_active", active)
                if active is False:
                        return
                
                if hasattr(p, 'incant_cfg_params') and 'scfg_params' in p.incant_cfg_params:
                        stats = p.incant_cfg_params['scfg_params'].statistics
                        logger.debug('SCFG Statistics: %s', stats)


                self.remove_all_hooks()

        def remove_all_hooks(self):
                all_crossattn_modules = self.get_all_crossattn_modules()
                for module in all_crossattn_modules:
                        self.remove_field_cross_attn_modules(module, 'scfg_last_to_q_map')
                        self.remove_field_cross_attn_modules(module, 'scfg_last_to_k_map')
                        if hasattr(module, 'to_q'):
                                handle_scfg_to_q = _remove_all_forward_hooks(module.to_q, 'scfg_to_q_hook')
                                self.remove_field_cross_attn_modules(module.to_q, 'scfg_parent_module')
                        if hasattr(module, 'to_k'):
                                handle_scfg_to_k = _remove_all_forward_hooks(module.to_k, 'scfg_to_k_hook')
                                self.remove_field_cross_attn_modules(module.to_k, 'scfg_parent_module')

        def unhook_callbacks(self, scfg_params: SCFGStateParams):
                pass

        def ready_hijack_forward(self, all_crossattn_modules):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                """

                def scfg_self_attn_hook(module, input, kwargs, output):
                        # scfg_q_map = output.detach().clone()
                        scfg_q_map = prepare_attn_map(output, module.scfg_heads)
                        attn_scores = get_attention_scores(scfg_q_map, scfg_q_map)
                        setattr(module.scfg_parent_module[0], 'scfg_last_qv_map', attn_scores)

                def scfg_cross_attn_hook(module, input, kwargs, output):
                        scfg_q_map = prepare_attn_map(module.scfg_parent_module[0].scfg_last_to_q_map, module.scfg_heads)
                        scfg_k_map = prepare_attn_map(output, module.scfg_heads)
                        #scfg_k_map = output.detach().clone()
                        attn_scores = get_attention_scores(scfg_q_map, scfg_k_map)
                        setattr(module.scfg_parent_module[0], 'scfg_last_qv_map', attn_scores)
                        # del module.parent_module[0].scfg_last_to_q_map

                def scfg_to_q_hook(module, input, kwargs, output):
                        setattr(module.scfg_parent_module[0], 'scfg_last_to_q_map', output)

                def scfg_to_k_hook(module, input, kwargs, output):
                        setattr(module.scfg_parent_module[0], 'scfg_last_to_k_map', output)

                for module in all_crossattn_modules:
                        if not hasattr(module, 'to_q') or not hasattr(module, 'to_k'):
                                logger.error("CrossAttention module '%s' does not have to_q or to_k", module.network_layer_name)
                                continue

                        # to_q
                        self.add_field_cross_attn_modules(module.to_q, 'scfg_parent_module', [module])
                        self.add_field_cross_attn_modules(module.to_q, 'scfg_last_to_q_map', None)
                        handle_scfg_to_q = module_hooks.module_add_forward_hook(
                                module.to_q,
                                scfg_to_q_hook,
                                with_kwargs=True
                        )

                        # to_k
                        self.add_field_cross_attn_modules(module.to_k, 'scfg_parent_module', [module])
                        if module.network_layer_name.endswith('attn2'): # cross attn
                                self.add_field_cross_attn_modules(module.to_k, 'scfg_last_to_k_map', None)
                                handle_scfg_to_k = module_hooks.module_add_forward_hook(
                                        module.to_k,
                                        scfg_to_k_hook,
                                        with_kwargs=True
                                )

        def get_all_crossattn_modules(self):
                """ 
                Get ALL attention modules
                """
                modules = module_hooks.get_modules(
                       module_name_filter='CrossAttention'
                )
                return modules

        def add_field_cross_attn_modules(self, module, field, value):
                """ Add a field to a module if it doesn't exist """
                module_hooks.modules_add_field(module, field, value)
        
        def remove_field_cross_attn_modules(self, module, field):
                """ Remove a field from a module if it exists """
                module_hooks.modules_remove_field(module, field)

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, scfg_params: SCFGStateParams):
                # always unhook
                self.unhook_callbacks(scfg_params)

        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, scfg_params: SCFGStateParams):
                """ Callback function for the CFGDenoisedParams 
                Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine
                
                """
                scfg_params.current_step = params.sampling_step

                # Run only within interval
                if not scfg_params.start_step <= params.sampling_step <= scfg_params.end_step:
                        return
                
                if scfg_params.scfg_scale <= 0:
                        return

                # S-CFG
                R = scfg_params.R
                max_latent_size = [params.x.shape[-2] // R, params.x.shape[-1] // R]

                #with LineProfiler(get_mask) as lp:
                ca_mask, fore_mask = get_mask(scfg_params.all_crossattn_modules,
                                        scfg_params,
                                        r = scfg_params.R,
                                        latent_size = max_latent_size,
                                )
                        #lp.print_stats()

                # todo parameterize this
                mask_t = F.interpolate(ca_mask, scale_factor=R, mode='nearest')
                mask_fore = F.interpolate(fore_mask, scale_factor=R, mode='nearest')
                scfg_params.mask_t = mask_t 
                scfg_params.mask_fore = mask_fore


        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[SCFG] Active", str, scfg_apply_override('scfg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SCFG] SCFG Scale", float, scfg_apply_field("scfg_scale")),
                        xyz_grid.AxisOption("[SCFG] SCFG Rate Min", float, scfg_apply_field("scfg_rate_min")),
                        xyz_grid.AxisOption("[SCFG] SCFG Rate Max", float, scfg_apply_field("scfg_rate_max")),
                        xyz_grid.AxisOption("[SCFG] SCFG Rate Clamp", float, scfg_apply_field("scfg_rate_clamp")),
                        xyz_grid.AxisOption("[SCFG] SCFG Start Step", int, scfg_apply_field("scfg_start_step")),
                        xyz_grid.AxisOption("[SCFG] SCFG End Step", int, scfg_apply_field("scfg_end_step")),
                        xyz_grid.AxisOption("[SCFG] SCFG R", int, scfg_apply_field("scfg_r")),
                }
                return extra_axis_options


def scfg_combine_denoised(model_delta, cfg_scale, scfg_params: SCFGStateParams):
        """ The inner loop of the S-CFG denoiser 
        Arguments:
                model_delta: torch.Tensor - defined by `x_out[cond_index] - denoised_uncond[i]`
                cfg_scale: float - guidance scale
                scfg_params: SCFGStateParams - the state parameters for the S-CFG denoiser
        
        Returns:
                int or torch.Tensor - 1.0 if not within interval or scale is 0, else the rate map tensor
        """

        current_step = scfg_params.current_step
        start_step = scfg_params.start_step
        end_step = scfg_params.end_step
        scfg_scale = scfg_params.scfg_scale

        if not start_step <= current_step <= end_step:
                return 1.0

        if scfg_scale <= 0:
                return 1.0

        mask_t = scfg_params.mask_t
        mask_fore = scfg_params.mask_fore
        min_rate = scfg_params.rate_min
        max_rate = scfg_params.rate_max
        rate_clamp = scfg_params.rate_clamp

        model_delta = model_delta.unsqueeze(0)
        model_delta_norm = model_delta.norm(dim=1, keepdim=True)

        eps = lambda dtype: torch.finfo(dtype).eps 

        # rescale map if necessary
        if mask_t.shape[2:] != model_delta_norm.shape[2:]:
                logger.debug('Rescaling mask_t from %s to %s', mask_t.shape[2:], model_delta_norm.shape[2:])
                mask_t = F.interpolate(mask_t, size=model_delta_norm.shape[2:], mode='bilinear')
        if mask_fore.shape[-2] != model_delta_norm.shape[-2]:
                logger.debug('Rescaling mask_fore from %s to %s', mask_fore.shape[2:], model_delta_norm.shape[2:])
                mask_fore = F.interpolate(mask_fore, size=model_delta_norm.shape[2:], mode='bilinear')

        delta_mask_norms = (model_delta_norm * mask_t).sum([2,3])/(mask_t.sum([2,3])+eps(mask_t.dtype))
        upnormmax = delta_mask_norms.max(dim=1)[0]
        upnormmax = upnormmax.unsqueeze(-1)

        fore_norms = (model_delta_norm * mask_fore).sum([2,3])/(mask_fore.sum([2,3])+eps(mask_fore.dtype))

        up = fore_norms
        down = delta_mask_norms

        tmp_mask = (mask_t.sum([2,3])>0).float()
        rate = up*(tmp_mask)/(down+eps(down.dtype)) # b 257
        rate = (rate.unsqueeze(-1).unsqueeze(-1)*mask_t).sum(dim=1, keepdim=True) # b 1, 64 64

        del model_delta_norm, delta_mask_norms, upnormmax, fore_norms, up, down, tmp_mask
        
        # unscaled min/max rate
        if rate.min().item() < scfg_params.statistics["min_rate"]:
                scfg_params.statistics["min_rate"] = rate.min().item()
        if rate.max().item() > scfg_params.statistics["max_rate"]:
                scfg_params.statistics["max_rate"] = rate.max().item()

        # should this go before or after the gaussian blur, or before/after the rate
        rate = rate * scfg_scale

        rate = torch.clamp(rate,min=min_rate, max=max_rate)

        if rate_clamp > 0:
                rate = torch.clamp_max(rate, rate_clamp/cfg_scale)

        ###Gaussian Smoothing 
        kernel_size = 3
        sigma=0.5
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(rate.device)
        rate = F.pad(rate, (1, 1, 1, 1), mode='reflect')
        rate = smoothing(rate)

        return rate.squeeze(0)


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def scfg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "scfg_active"):
                setattr(p, "scfg_active", True)
    return fun


def scfg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "scfg_active"):
                setattr(p, "scfg_active", True)
        setattr(p, field, x)
    return fun


def _remove_all_forward_hooks(
    module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> None:
        module_hooks.remove_module_forward_hook(module, hook_fn_name)


"""
# below code modified from https://github.com/SmilesDZgk/S-CFG
@inproceedings{shen2024rethinking,
  title={Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidancee},
  author={Shen, Dazhong and Song, Guanglu and Xue, Zeyue and Wang, Fu-Yun and Liu, Yu},
  booktitle={Proceedings of The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024}
}
"""


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)

# based on diffusers/models/attention_processor.py Attention head_to_batch_dim
def head_to_batch_dim(x, heads, out_dim=3):
        head_size = heads
        if x.ndim == 3:

                batch_size, seq_len, dim = x.shape
                extra_dim = 1
        else:
               batch_size, extra_dim, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        x = x.permute(0, 2, 1, 3)
        if out_dim == 3:
               x = x.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)
        return x


# based on diffusers/models/attention_processor.py Attention batch_to_head_dim
def batch_to_head_dim(x, heads):
        head_size = heads
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // head_size, head_size, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return x


def average_over_head_dim(x, heads):
        x = rearrange(x, '(b h) s t -> b h s t', h=heads).mean(1)
        return x


import torch.nn.functional as F
from einops import rearrange
def get_mask(attn_modules, scfg_params: SCFGStateParams, r, latent_size):
        """ Aggregates the attention across the different layers and heads at the specified resolution. 
        In the original paper, r is a hyper-parameter set to 4.
        Arguments:
                attn_modules: List of attention modules
                scfg_params: SCFGStateParams
                r: int - 
                latent_size: tuple 
        
        """
        height = scfg_params.height
        width = scfg_params.width
        max_dims = height * width
        latent_size = latent_size[-2:]
        module_attn_sizes = set()

        key_corss = f"r{r}_cross"
        key_self = f"r{r}_self"


        # The maximum value of the sizes of attention map to aggregate
        max_r = r
        max_sizes = r

        # The current number of attention map resolutions aggregated
        attnmap_r = 0

        r_r = 1
        new_ca = 0
        new_fore=0
        a_n=0
        # corresponds to diffusers pipe.unet.config.sample_size
        # sample_size = 64
        # get a layer wise mapping
        attention_store_proxy = {"r2_cross": [], "r4_cross": [], "r8_cross": [], "r16_cross": [],
                                 "r2_self": [], "r4_self": [], "r8_self": [], "r16_self": []}
        for module in attn_modules:
                module_type = 'cross' if 'attn2' in module.network_layer_name else 'self'

                to_q_map = getattr(module, 'scfg_last_to_q_map', None)
                to_k_map = getattr(module, 'scfg_last_to_k_map', None)
                # self-attn
                if to_k_map is None:
                        to_k_map = to_q_map

                to_q_map = prepare_attn_map(to_q_map, module.heads)
                to_k_map = prepare_attn_map(to_k_map, module.heads)

                module_attn_size = to_q_map.size(1)
                module_attn_sizes.add(module_attn_size)
                downscale_h = int((module_attn_size * (height / width)) ** 0.5)
                downscale_w = module_attn_size // downscale_h
                module_key = f"r{module_attn_size}_{module_type}"

                attn_probs = get_attention_scores(to_q_map, to_k_map, to_q_map.dtype)

                if module_type == 'self':
                       del module.scfg_last_to_q_map
                else:
                       del module.scfg_last_to_q_map, module.scfg_last_to_k_map

                if module_key not in attention_store_proxy:
                        attention_store_proxy[module_key] = []
                try:
                        attention_store_proxy[module_key].append(attn_probs)
                except KeyError:
                        continue

        module_attn_sizes = sorted(list(module_attn_sizes))
        attention_maps = attention_store_proxy

        curr_r = module_attn_sizes.pop(0)
        while curr_r != None and attnmap_r < max_sizes:
                key_corss = f"r{curr_r}_cross"
                key_self = f"r{curr_r}_self"

                if key_self not in attention_maps.keys() or key_corss not in attention_maps.keys():
                        next_r = module_attn_sizes.pop(0)
                        attnmap_r += 1
                        curr_r = next_r
                        continue
                if len(attention_maps[key_self]) == 0 or len(attention_maps[key_corss]) == 0:
                        curr_r = module_attn_sizes.pop(0)
                        attnmap_r += 1
                        curr_r = next_r
                        continue

                sa = torch.stack(attention_maps[key_self], dim=1)
                ca = torch.stack(attention_maps[key_corss], dim=1)
                attn_num = sa.size(1)
                sa = rearrange(sa, 'b n h w -> (b n) h w')
                ca = rearrange(ca, 'b n h w -> (b n) h w')

                curr = 0 # b hw c=hw
                curr +=sa

                # 4.1.2 Self-Attentiion
                ssgc_sa = curr
                ssgc_n = max_r

                # summation from r=2 to R, we set ssgc_sa to curr which would be sa^1
                # major memory hog
                #    active_bytes peak from 3.41G to 4.04G
                #    reserved_bytes peak from 3.70G to 4.64G
                # optimization 1: active 4.03G -> 3.72G = 0.31G, reserved 4.64G -> 4.16G = 0.48G
                for r_value in range(1, ssgc_n):
                        r_pow = r_value + 1
                        curr @= sa  # optimization 1
#                        curr = torch.linalg.matrix_power(sa, r_pow) # sa^r
                        ssgc_sa += curr

                ssgc_sa/=ssgc_n
                sa = ssgc_sa

                ########smoothing ca
                ca = sa@ca # b hw c

                hw = ca.size(1)

                downscale_h = round((hw * (height / width)) ** 0.5)

                ca = rearrange(ca, 'b (h w) c -> b c h w', h=downscale_h )

                # Scale the attention map to the expected size
                max_size = latent_size
                scale_factor = [
                        max_size[0] / ca.shape[-2],
                        max_size[1] / ca.shape[-1]
                ]
                mode =  'bilinear' #'nearest' #
                ca = F.interpolate(ca, scale_factor=scale_factor, mode=mode) # b 77 32 32

                #####Gaussian Smoothing
                kernel_size = 3
                sigma = 0.5
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(ca.device)
                channel = ca.size(1)
                ca= rearrange(ca, ' b c h w -> (b c) h w' ).unsqueeze(1)
                ca = F.pad(ca, (1, 1, 1, 1), mode='reflect')
                ca = smoothing(ca.float()).squeeze(1)
                ca = rearrange(ca, ' (b c) h w -> b c h w' , c= channel)
                
                ca_norm = ca/(ca.mean(dim=[2,3], keepdim=True)+torch.finfo(ca.dtype).eps) ### spatial  normlization 
               
                new_ca+=rearrange(ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1) 

                fore_ca = torch.stack([ca[:,0],ca[:,1:].sum(dim=1)], dim=1)
                froe_ca_norm = fore_ca/fore_ca.mean(dim=[2,3], keepdim=True) ### spatial  normlization 
                new_fore += rearrange(froe_ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1)  
                a_n+=attn_num

                if len(module_attn_sizes) > 0:
                        curr_r = module_attn_sizes.pop(0)
                else:
                        curr_r = None
                attnmap_r += 1
                # r_r *= 2

                # optimization 2: memory savings: 3.09G - 2.47G = 0.62G
                del ca_norm, froe_ca_norm, fore_ca

        # no memory savings
        del attention_maps
        del sa, ca, ssgc_sa, ssgc_n, curr
        
        # variables used from above:
        # new_ca, new_fore, a_n
        new_ca = new_ca/a_n
        new_fore = new_fore/a_n
        _,new_ca   = new_ca.chunk(2, dim=0) #[1]
        fore_ca, _ = new_fore.chunk(2, dim=0)

        max_ca, inds = torch.max(new_ca[:,:], dim=1) 
        max_ca = max_ca.unsqueeze(1) # 
        ca_mask = (new_ca==max_ca).float() # b 77/10 16 16 

        max_fore, inds = torch.max(fore_ca[:,:], dim=1) 
        max_fore = max_fore.unsqueeze(1) # 
        fore_mask = (fore_ca==max_fore).float() # b 77/10 16 16 
        fore_mask = 1.0-fore_mask[:,:1] # b 1 16 16

        # no memory savings
        del new_ca, new_fore, a_n, max_ca, max_fore, inds

        return [ ca_mask, fore_mask]


def prepare_attn_map(to_k_map, heads):
    to_k_map = head_to_batch_dim(to_k_map, heads)
    to_k_map = average_over_head_dim(to_k_map, heads)
    to_k_map = torch.stack([to_k_map[0], to_k_map[0]], dim=0)
    return to_k_map


def get_attention_scores(to_q_map, to_k_map, dtype):
        """ Calculate the attention scores for the given query and key maps
        Arguments:
                to_q_map: torch.Tensor - query map
                to_k_map: torch.Tensor - key map
                dtype: torch.dtype - data type of the tensor
        Returns:
                torch.Tensor - attention scores 
        """
        # based on diffusers models/attention.py "get_attention_scores"
        # use in place operations vs. softmax to save memory: https://stackoverflow.com/questions/53732209/torch-in-place-operations-to-save-memory-softmax
        # 512x: 2.65G -> 2.47G
        # attn_probs = attn_scores.softmax(dim=-1).to(device=shared.device, dtype=to_q_map.dtype)

        attn_probs = to_q_map @ to_k_map.transpose(-1, -2)

        # avoid nan by converting to float32 and subtracting max 
        attn_probs = attn_probs.to(dtype=torch.float32) #
        attn_probs -= torch.max(attn_probs)

        torch.exp(attn_probs, out = attn_probs)
        summed = attn_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        attn_probs /= summed

        attn_probs = attn_probs.to(dtype=dtype)

        return attn_probs