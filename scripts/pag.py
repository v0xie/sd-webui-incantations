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
An unofficial implementation of "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance" for Automatic1111 WebUI.

@misc{ahn2024selfrectifying,
      title={Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance}, 
      author={Donghoon Ahn and Hyoungwon Cho and Jaewon Min and Wooseok Jang and Jungwoo Kim and SeonHwa Kim and Hyun Hee Park and Kyong Hwan Jin and Seungryong Kim},
      year={2024},
      eprint={2403.17377},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Include noise interval for CFG and PAG guidance in the sampling process from "Applying Guidance in a Limited Interval Improves
Sample and Distribution Quality in Diffusion Models"

@misc{kynkäänniemi2024applying,
      title={Applying Guidance in a Limited Interval Improves Sample and Distribution Quality in Diffusion Models}, 
      author={Tuomas Kynkäänniemi and Miika Aittala and Tero Karras and Samuli Laine and Timo Aila and Jaakko Lehtinen},
      year={2024},
      eprint={2404.07724},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Include CFG schedulers from "Analysis of Classifier-Free Guidance Weight Schedulers"

@misc{wang2024analysis,
      title={Analysis of Classifier-Free Guidance Weight Schedulers}, 
      author={Xi Wang and Nicolas Dufour and Nefeli Andreou and Marie-Paule Cani and Victoria Fernandez Abrevaya and David Picard and Vicky Kalogeiton},
      year={2024},
      eprint={2404.13040},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


handles = []
global_scale = 1

SCHEDULES = [
        'Constant',
        'Clamp-Linear (c=4.0)',
        'Clamp-Linear (c=2.0)',
        'Clamp-Linear (c=1.0)',
        'Linear',
        'Inverse-Linear',
        'Cosine',
        'Clamp-Cosine (c=4.0)',
        'Clamp-Cosine (c=2.0)',
        'Clamp-Cosine (c=1.0)',
        'Sine',
        'Interval',
        'PCS (s=0.01)',
        'PCS (s=0.1)',
        'PCS (s=1.0)',
        'PCS (s=2.0)',
        'PCS (s=4.0)',
]

SCFG_MODULES = ['to_q', 'to_v', 'to_k']



class PAGStateParams:
        def __init__(self):
                self.pag_scale: int = -1      # PAG guidance scale
                self.pag_start_step: int = 0
                self.pag_end_step: int = 150 
                self.cfg_interval_enable: bool = False
                self.cfg_interval_schedule: str = 'Constant'
                self.cfg_interval_low: float = 0
                self.cfg_interval_high: float = 50.0
                self.step : int = 0 
                self.max_sampling_step : int = 1 
                self.guidance_scale: int = -1 # CFG
                self.x_in = None
                self.text_cond = None
                self.image_cond = None
                self.sigma = None
                self.text_uncond = None
                self.make_condition_dict = None # callable lambda
                self.crossattn_modules = []
                self.all_crossattn_modules = []
                self.to_v_modules = []
                self.to_out_modules = []
                self.pag_x_out = None
                self.batch_size = -1      # Batch size
                self.denoiser = None # CFGDenoiser
                self.patched_combine_denoised = None
                self.conds_list = None
                self.uncond_shape_0 = None

class SCFGStateParams:
        def __init__(self):
                self.all_crossattn_modules = []
                self.max_out_dim = 1280
                self.mask_t = None
                self.mask_fore = None


class PAGExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "Perturbed Attention Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Perturbed Attention Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='pag_active')
                        with gr.Row():
                                pag_scale = gr.Slider(value = 0, minimum = 0, maximum = 20.0, step = 0.5, label="PAG Scale", elem_id = 'pag_scale', info="")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'pag_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'pag_end_step', info="")
                        with gr.Row():
                                cfg_interval_enable = gr.Checkbox(value=False, default=False, label="Enable CFG Scheduler", elem_id='cfg_interval_enable', info="If enabled, applies CFG only within noise interval with the selected schedule type. PAG must be enabled (scale can be 0). SDXL recommend CFG=15; CFG interval (0.28, 5.42]")
                                cfg_schedule = gr.Dropdown(
                                        value='Constant',
                                        choices= SCHEDULES,
                                        label="CFG Schedule Type", 
                                        elem_id='cfg_interval_schedule', 
                                )
                                with gr.Row():
                                        cfg_interval_low = gr.Slider(value = 0, minimum = 0, maximum = 100, step = 0.01, label="CFG Noise Interval Low", elem_id = 'cfg_interval_low', info="")
                                        cfg_interval_high = gr.Slider(value = 100, minimum = 0, maximum = 100, step = 0.01, label="CFG Noise Interval High", elem_id = 'cfg_interval_high', info="")
                                
                active.do_not_save_to_config = True
                pag_scale.do_not_save_to_config = True
                start_step.do_not_save_to_config = True
                end_step.do_not_save_to_config = True
                cfg_interval_enable.do_not_save_to_config = True
                cfg_schedule.do_not_save_to_config = True
                cfg_interval_low.do_not_save_to_config = True
                cfg_interval_high.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='PAG Active' in d)),
                        (pag_scale, 'PAG Scale'),
                        (start_step, 'PAG Start Step'),
                        (end_step, 'PAG End Step'),
                        (cfg_interval_enable, 'CFG Interval Enable'),
                        (cfg_schedule, 'CFG Interval Schedule'),
                        (cfg_interval_low, 'CFG Interval Low'),
                        (cfg_interval_high, 'CFG Interval High')
                ]
                self.paste_field_names = [
                        'pag_active',
                        'pag_scale',
                        'pag_start_step',
                        'pag_end_step',
                        'cfg_interval_enable',
                        'cfg_interval_schedule',
                        'cfg_interval_low',
                        'cfg_interval_high',
                ]
                return [active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.pag_process_batch(p, *args, **kwargs)

        def pag_process_batch(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "pag_active", active)
                if active is False:
                        return
                pag_scale = getattr(p, "pag_scale", pag_scale)
                start_step = getattr(p, "pag_start_step", start_step)
                end_step = getattr(p, "pag_end_step", end_step)

                cfg_interval_enable = getattr(p, "cfg_interval_enable", cfg_interval_enable)
                cfg_schedule = getattr(p, "cfg_interval_schedule", cfg_schedule)
                cfg_interval_low = getattr(p, "cfg_interval_low", cfg_interval_low)
                cfg_interval_high = getattr(p, "cfg_interval_high", cfg_interval_high)

                p.extra_generation_params.update({
                        "PAG Active": active,
                        "PAG Scale": pag_scale,
                        "PAG Start Step": start_step,
                        "PAG End Step": end_step,
                        "CFG Interval Enable": cfg_interval_enable,
                        "CFG Interval Schedule": cfg_schedule,
                        "CFG Interval Low": cfg_interval_low,
                        "CFG Interval High": cfg_interval_high
                })
                self.create_hook(p, active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high)

        def create_hook(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, *args, **kwargs):
                # Create a list of parameters for each concept
                pag_params = PAGStateParams()
                pag_params.pag_scale = pag_scale
                pag_params.pag_start_step = start_step
                pag_params.pag_end_step = end_step
                pag_params.cfg_interval_enable = cfg_interval_enable
                pag_params.cfg_interval_schedule = cfg_schedule
                pag_params.max_sampling_step = p.steps
                pag_params.guidance_scale = p.cfg_scale
                pag_params.batch_size = p.batch_size
                pag_params.denoiser = None

                if pag_params.cfg_interval_enable:
                       # Refer to 3.1 Practice in the paper
                       # We want to round high and low noise levels to the nearest integer index
                       low_index = find_closest_index(cfg_interval_low, pag_params.max_sampling_step)
                       high_index = find_closest_index(cfg_interval_high, pag_params.max_sampling_step)
                       pag_params.cfg_interval_low = calculate_noise_level(low_index, pag_params.max_sampling_step)
                       pag_params.cfg_interval_high = calculate_noise_level(high_index, pag_params.max_sampling_step)
                       logger.debug(f"Step Aligned CFG Interval (low, high): ({low_index}, {high_index}), Step Aligned CFG Interval: ({round(pag_params.cfg_interval_low, 4)}, {round(pag_params.cfg_interval_high, 4)})")

                # Get all the qv modules
                scfg_params = SCFGStateParams()
                all_crossattn_modules = self.get_all_crossattn_modules()
                if len(all_crossattn_modules) > 0:
                        scfg_params.all_crossattn_modules = all_crossattn_modules

                        # Get the max out dim to upscale targeted crossattn maps
                        scfg_params.max_out_dim = max([m.to_v.out_features for m in all_crossattn_modules])

                cross_attn_modules = self.get_cross_attn_modules()
                if len(cross_attn_modules) == 0:
                        logger.error("No cross attention modules found, cannot proceed with PAG")
                        return
                pag_params.crossattn_modules = [m for m in cross_attn_modules if 'CrossAttention' in m.__class__.__name__]

                # Use lambda to call the callback function with the parameters to avoid global variables
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, pag_params, scfg_params)
                cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(callback_params, pag_params, scfg_params)
                #after_cfg_lambda = lambda x: self.cfg_after_cfg_callback(x, params)
                unhook_lambda = lambda _: self.unhook_callbacks(pag_params)

                self.ready_hijack_forward(pag_params.crossattn_modules, scfg_params.all_crossattn_modules, pag_scale)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_cfg_denoised(cfg_denoised_lambda)
                #script_callbacks.on_cfg_after_cfg(after_cfg_lambda)
                script_callbacks.on_script_unloaded(unhook_lambda)

        def postprocess_batch(self, p, *args, **kwargs):
                self.pag_postprocess_batch(p, *args, **kwargs)

        def pag_postprocess_batch(self, p, active, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()

                logger.debug('Removed script callbacks')
                active = getattr(p, "pag_active", active)
                if active is False:
                        return

        def remove_all_hooks(self):
                cross_attn_modules = self.get_cross_attn_modules()
                for module in cross_attn_modules:
                        to_v = getattr(module, 'to_v', None)
                        self.remove_field_cross_attn_modules(module, 'pag_enable')
                        self.remove_field_cross_attn_modules(module, 'pag_last_to_v')
                        self.remove_field_cross_attn_modules(to_v, 'pag_parent_module')
                        _remove_all_forward_hooks(module, 'pag_pre_hook')
                        _remove_all_forward_hooks(to_v, 'to_v_pre_hook')
                        #_remove_all_forward_hooks(to_v, 'scfg_to_v_hook')

                all_crossattn_modules = self.get_all_crossattn_modules()
                for module in all_crossattn_modules:
                        self.remove_field_cross_attn_modules(module, 'scfg_last_attn_map')
                        self.remove_field_cross_attn_modules(module, 'scfg_last_context_map')
                        self.remove_field_cross_attn_modules(module, 'scfg_last_to_q_map')
                        self.remove_field_cross_attn_modules(module, 'scfg_last_to_k_map')
                        self.remove_field_cross_attn_modules(module, 'scfg_last_to_v_map')
                        self.remove_field_cross_attn_modules(module, 'scfg_last_to_out_map')
                        self.remove_field_cross_attn_modules(module, 'scfg_attn_size')
                        _remove_all_forward_hooks(module, 'scfg_hook')
                        _remove_all_forward_hooks(module, 'scfg_pre_hook')
                        to_v = getattr(module, 'to_v', None)
                        _remove_all_forward_hooks(to_v, 'scfg_to_v_hook')
                        if hasattr(module, 'to_q'):
                                handle_scfg_to_q = _remove_all_forward_hooks(module.to_q, 'scfg_to_q_hook')
                                self.remove_field_cross_attn_modules(module.to_q, 'scfg_parent_module')
                        if hasattr(module, 'to_k'):
                                handle_scfg_to_q = _remove_all_forward_hooks(module.to_k, 'scfg_to_k_hook')
                                self.remove_field_cross_attn_modules(module.to_k, 'scfg_parent_module')
                        if hasattr(module, 'to_v'):
                                handle_scfg_to_v = _remove_all_forward_hooks(module.to_v, 'scfg_to_v_hook')
                                self.remove_field_cross_attn_modules(module.to_v, 'scfg_parent_module')
                        if hasattr(module, 'to_out'):
                                handle_scfg_to_out = _remove_all_forward_hooks(module.to_out[0], 'scfg_to_out_hook')
                                self.remove_field_cross_attn_modules(module.to_out[0], 'scfg_parent_module')

        def unhook_callbacks(self, pag_params: PAGStateParams):
                global handles

                if pag_params is None:
                       logger.error("PAG params is None")
                       return

                if pag_params.denoiser is not None:
                        denoiser = pag_params.denoiser
                        setattr(denoiser, 'combine_denoised_patched', False)
                        try:
                                patches.undo(__name__, denoiser, "combine_denoised")
                        except KeyError:
                                logger.exception("KeyError unhooking combine_denoised")
                                pass
                        except RuntimeError:
                                logger.exception("RuntimeError unhooking combine_denoised")
                                pass
                        pag_params.denoiser = None


        def ready_hijack_forward(self, crossattn_modules, all_crossattn_modules, pag_scale):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                Then applies the PAG perturbation to the output of the cross attention module (multiplication by identity)
                """

                # add field for last_to_v
                for module in crossattn_modules:
                        to_v = getattr(module, 'to_v', None)
                        self.add_field_cross_attn_modules(module, 'pag_enable', False)
                        self.add_field_cross_attn_modules(module, 'pag_last_to_v', None)
                        self.add_field_cross_attn_modules(to_v, 'pag_parent_module', [module])
                        self.add_field_cross_attn_modules(module, 'scfg_last_attn_map', None)
                        self.add_field_cross_attn_modules(module, 'scfg_attn_size', -1)
                        # self.add_field_cross_attn_modules(to_out, 'pag_parent_module', [module])
                        
                # add field for last_to_v
                for module in all_crossattn_modules:
                        self.add_field_cross_attn_modules(module, 'scfg_last_attn_map', None)
                        self.add_field_cross_attn_modules(module, 'scfg_last_context_map', None)
                        self.add_field_cross_attn_modules(module, 'scfg_last_to_q_map', None)
                        self.add_field_cross_attn_modules(module, 'scfg_last_to_k_map', None)
                        self.add_field_cross_attn_modules(module, 'scfg_last_to_v_map', None)
                        self.add_field_cross_attn_modules(module, 'scfg_last_to_out_map', None)
                        self.add_field_cross_attn_modules(module, 'scfg_attn_size', -1)
                        for submodule in SCFG_MODULES:
                                sub_module = getattr(module, submodule, None)
                                self.add_field_cross_attn_modules(sub_module, 'scfg_parent_module', [module])
                        if hasattr(module, 'to_out'):
                                self.add_field_cross_attn_modules(module.to_out[0], 'scfg_parent_module', [module])

                def to_v_pre_hook(module, input, kwargs, output):
                        """ Copy the output of the to_v module to the parent module """
                        parent_module = getattr(module, 'pag_parent_module', None)
                        # copy the output of the to_v module to the parent module
                        setattr(parent_module[0], 'pag_last_to_v', output.detach().clone())

                def scfg_to_q_hook(module, input, kwargs, output):
                        setattr(module.scfg_parent_module[0], 'scfg_last_to_q_map', output.detach().clone())

                def scfg_to_k_hook(module, input, kwargs, output):
                        setattr(module.scfg_parent_module[0], 'scfg_last_to_k_map', output.detach().clone())

                def scfg_to_v_hook(module, input, kwargs, output):
                        setattr(module.scfg_parent_module[0], 'scfg_last_to_v_map', output.detach().clone())

                def scfg_to_out_hook(module, input, kwargs, output):
                        setattr(module.scfg_parent_module[0], 'scfg_last_to_out_map', output.detach().clone())

                def pag_pre_hook(module, input, kwargs, output):
                        if hasattr(module, 'pag_enable') and getattr(module, 'pag_enable', False) is False:
                                return
                        if not hasattr(module, 'pag_last_to_v'):
                                # oops we forgot to unhook
                                return

                        batch_size, seq_len, inner_dim = output.shape
                        identity = torch.eye(seq_len).expand(batch_size, -1, -1).to(shared.device)

                        # get the last to_v output and save it
                        last_to_v = getattr(module, 'pag_last_to_v', None)
                        if last_to_v is not None:    
                                new_output = torch.einsum('bij,bjk->bik', identity, last_to_v)
                                return new_output
                        else:
                                # this is bad
                                return output

                def scfg_pre_hook(module, args, kwargs):
                        if not hasattr(module, 'scfg_last_attn_map'):
                                return
                        #if kwargs.get('context', None) is not None:
                        #        setattr(module, 'scfg_last_context_map', kwargs.get('context').detach().clone())
                        #setattr(module, 'scfg_last_attn_map', args[0].detach().clone())

                def scfg_hook(module, input, kwargs, output):
                        if not hasattr(module, 'scfg_last_attn_map'):
                                return
                        if kwargs.get('context', None) is not None:
                                setattr(module, 'scfg_last_context_map', kwargs.get('context').detach().clone())
                                #to_v_map = getattr(module, 'scfg_last_to_v_map', None)
                                #attn_map = (input[0] @ to_v_map.transpose(1,2)).transpose(1,2)
                        #        setattr(module, 'scfg_last_attn_map', attn_map)
                        #else:
                        setattr(module, 'scfg_last_attn_map', output.detach().clone())
                        #if kwargs.get('context', None) is not None:
                        #        setattr(module, 'scfg_last_attn_map', kwargs.get('context').detach().clone())
                        #else:
                        #setattr(module, 'scfg_last_attn_map', output.detach().clone())
                        # setattr(module, 'scfg_attn_size', output.size(1) ** 0.5)


                # Create hooks 
                for module in crossattn_modules:
                        handle_parent = module.register_forward_hook(pag_pre_hook, with_kwargs=True)
                        to_v = getattr(module, 'to_v', None)
                        handle_to_v = to_v.register_forward_hook(to_v_pre_hook, with_kwargs=True)
                
                for module in all_crossattn_modules:
                        handle_scfg = module.register_forward_hook(scfg_hook, with_kwargs=True)
                        handle_scfg_pre = module.register_forward_pre_hook(scfg_pre_hook, with_kwargs=True)
                        if hasattr(module, 'to_q'):
                                handle_scfg_to_q = module.to_q.register_forward_hook(scfg_to_q_hook, with_kwargs=True)
                        if hasattr(module, 'to_k'):
                                handle_scfg_to_k = module.to_k.register_forward_hook(scfg_to_k_hook, with_kwargs=True)
                        if hasattr(module, 'to_v'):
                                handle_scfg_to_v= module.to_v.register_forward_hook(scfg_to_v_hook, with_kwargs=True)
                        if hasattr(module, 'to_out'):
                                handle_scfg_to_out = module.to_out[0].register_forward_hook(scfg_to_out_hook, with_kwargs=True)

        def get_all_crossattn_modules(self):
                """ 
                Get ALL attention modules
                """
                try:
                        m = shared.sd_model
                        nlm = m.network_layer_mapping
                        middle_block_modules = [m for m in nlm.values() if 'CrossAttention' in m.__class__.__name__]
                        return middle_block_modules
                except AttributeError:
                        logger.exception("AttributeError in get_middle_block_modules", stack_info=True)
                        return []
                except Exception:
                        logger.exception("Exception in get_middle_block_modules", stack_info=True)
                        return []

        def get_middle_block_modules(self):
                """ Get all attention modules from the middle block 
                Refere to page 22 of the PAG paper, Appendix A.2
                
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

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: PAGStateParams, scfg_params: SCFGStateParams):
                # always unhook
                self.unhook_callbacks(pag_params)

                pag_params.step = params.sampling_step

                # patch combine_denoised
                if pag_params.denoiser is None:
                        pag_params.denoiser = params.denoiser
                if getattr(params.denoiser, 'combine_denoised_patched', False) is False:
                        try:
                                setattr(params.denoiser, 'combine_denoised_original', params.denoiser.combine_denoised)
                                # create patch that references the original function
                                pass_conds_func = lambda *args, **kwargs: combine_denoised_pass_conds_list(
                                        *args,
                                        **kwargs,
                                        original_func = params.denoiser.combine_denoised_original,
                                        pag_params = pag_params,
                                        scfg_params = scfg_params)
                                pag_params.patched_combine_denoised = patches.patch(__name__, params.denoiser, "combine_denoised", pass_conds_func)
                                setattr(params.denoiser, 'combine_denoised_patched', True)
                                setattr(params.denoiser, 'combine_denoised_original', patches.original(__name__, params.denoiser, "combine_denoised"))
                        except KeyError:
                                logger.exception("KeyError patching combine_denoised")
                                pass
                        except RuntimeError:
                                logger.exception("RuntimeError patching combine_denoised")
                                pass

                # Run PAG only within interval
                if pag_params.pag_start_step <= params.sampling_step <= pag_params.pag_end_step and pag_params.pag_scale > 0:
                        if isinstance(params.text_cond, dict):
                                text_cond = params.text_cond['crossattn'] # SD XL
                                pag_params.text_cond = {}
                                pag_params.text_uncond = {}
                                for key, value in params.text_cond.items():
                                        pag_params.text_cond[key] = value.clone().detach()
                                        pag_params.text_uncond[key] = value.clone().detach()
                        else:
                                text_cond = params.text_cond # SD 1.5
                                pag_params.text_cond = text_cond.clone().detach()
                                pag_params.text_uncond = text_cond.clone().detach()

                        pag_params.x_in = params.x.clone().detach()
                        pag_params.sigma = params.sigma.clone().detach()
                        pag_params.image_cond = params.image_cond.clone().detach()
                        pag_params.denoiser = params.denoiser
                        pag_params.make_condition_dict = get_make_condition_dict_fn(params.text_uncond)




        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, pag_params: PAGStateParams, scfg_params: SCFGStateParams):
                """ Callback function for the CFGDenoisedParams 
                Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine
                
                """
                # S-CFG
                ca_mask, fore_mask = get_mask(scfg_params.all_crossattn_modules)

                # todo parameterize this
                R = 4
                mask_t = F.interpolate(ca_mask, scale_factor=R, mode='nearest')
                mask_fore = F.interpolate(fore_mask, scale_factor=R, mode='nearest')
                scfg_params.mask_t = mask_t
                scfg_params.mask_fore = mask_fore

                # Run only within interval
                if not pag_params.pag_start_step <= params.sampling_step <= pag_params.pag_end_step or pag_params.pag_scale <= 0:
                        return

                # passed from on_cfg_denoiser_callback
                x_in = pag_params.x_in
                tensor = pag_params.text_cond
                uncond = pag_params.text_uncond
                image_cond_in = pag_params.image_cond
                sigma_in = pag_params.sigma
                
                # concatenate the conditions 
                # "modules/sd_samplers_cfg_denoiser.py:237"
                cond_in = catenate_conds([tensor, uncond])
                make_condition_dict = get_make_condition_dict_fn(uncond)
                conds = make_condition_dict(cond_in, image_cond_in)
                
                # set pag_enable to True for the hooked cross attention modules
                for module in pag_params.crossattn_modules:
                        setattr(module, 'pag_enable', True)

                # get the PAG guidance (is there a way to optimize this so we don't have to calculate it twice?)
                pag_x_out = params.inner_model(x_in, sigma_in, cond=conds)

                # update pag_x_out
                pag_params.pag_x_out = pag_x_out

                # set pag_enable to False
                for module in pag_params.crossattn_modules:
                        setattr(module, 'pag_enable', False)
        
        def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams, pag_params: PAGStateParams):
                #self.unhook_callbacks(pag_params)
                pass

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[PAG] Active", str, pag_apply_override('pag_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] PAG Scale", float, pag_apply_field("pag_scale")),
                        xyz_grid.AxisOption("[PAG] PAG Start Step", int, pag_apply_field("pag_start_step")),
                        xyz_grid.AxisOption("[PAG] PAG End Step", int, pag_apply_field("pag_end_step")),
                        xyz_grid.AxisOption("[PAG] Enable CFG Scheduler", str, pag_apply_override('cfg_interval_enable', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] CFG Noise Interval Low", float, pag_apply_field("cfg_interval_low")),
                        xyz_grid.AxisOption("[PAG] CFG Noise Interval High", float, pag_apply_field("cfg_interval_high")),
                        xyz_grid.AxisOption("[PAG] CFG Schedule Type", str, pag_apply_override('cfg_interval_schedule', boolean=False), choices=lambda: SCHEDULES),
                        #xyz_grid.AxisOption("[PAG] ctnms_alpha", float, pag_apply_field("pag_ctnms_alpha")),
                }
                return extra_axis_options


def combine_denoised_pass_conds_list(*args, **kwargs):
        """ Hijacked function for combine_denoised in CFGDenoiser """
        original_func = kwargs.get('original_func', None)
        new_params = kwargs.get('pag_params', None)
        scfg_params = kwargs.get('scfg_params', None)

        if new_params is None:
                logger.error("new_params is None")
                return original_func(*args)

        def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
                denoised_uncond = x_out[-uncond.shape[0]:]
                denoised = torch.clone(denoised_uncond)

                noise_level = calculate_noise_level(new_params.step, new_params.max_sampling_step)

                # Calculate CFG Scale
                cfg_scale = cond_scale
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

                if incantations_debug:
                        logger.debug(f"Schedule: {new_params.cfg_interval_schedule}, CFG Scale: {cfg_scale}, Noise_level: {round(noise_level,3)}")

                for i, conds in enumerate(conds_list):
                        for cond_index, weight in conds:
                                if scfg_params is not None:
                                        mask_t = scfg_params.mask_t
                                        mask_fore = scfg_params.mask_fore

                                        model_delta = (x_out[cond_index] - denoised_uncond[i]).unsqueeze(0)
                                        model_delta_norm = model_delta.norm(dim=1, keepdim=True)
                                        delta_mask_norms = (model_delta_norm * scfg_params.mask_t).sum([2,3])/(mask_t.sum([2,3])+1e-8)
                                        upnormmax = delta_mask_norms.max(dim=1)[0]
                                        upnormmax = upnormmax.unsqueeze(-1)

                                        fore_norms = (model_delta_norm * mask_fore).sum([2,3])/(mask_fore.sum([2,3])+1e-8)

                                        up = fore_norms
                                        down = delta_mask_norms
                                        

                                        tmp_mask = (mask_t.sum([2,3])>0).float()
                                        rate = up*(tmp_mask)/(down+1e-8) # b 257
                                        rate = (rate.unsqueeze(-1).unsqueeze(-1)*mask_t).sum(dim=1, keepdim=True) # b 1, 64 64
                                        
                                        rate = torch.clamp(rate,min=0.8, max=3.0)
                                        rate = torch.clamp_max(rate, 15.0/cfg_scale)

                                        ###Gaussian Smoothing 
                                        kernel_size = 3
                                        sigma=0.5
                                        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(rate.device)
                                        rate = F.pad(rate, (1, 1, 1, 1), mode='reflect')
                                        rate = smoothing(rate)
                                        rate = rate.to(x_out[cond_index].dtype)
                                        denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * rate.squeeze(0) * (weight * cfg_scale)

                                else:
                                        denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cfg_scale)

                                # Apply PAG guidance only within interval
                                if not new_params.pag_start_step <= new_params.step <= new_params.pag_end_step or new_params.pag_scale <= 0:
                                        continue
                                else:
                                        try:
                                                denoised[i] += (x_out[cond_index] - new_params.pag_x_out[i]) * (weight * new_params.pag_scale)
                                        except TypeError:
                                                logger.exception("TypeError in combine_denoised_pass_conds_list")
                                        except IndexError:
                                                logger.exception("IndexError in combine_denoised_pass_conds_list")
                                        #logger.debug(f"added PAG guidance to denoised - pag_scale:{global_scale}")
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


def calculate_noise_level(i, N, sigma_min=0.002, sigma_max=80.0, rho=3):
    """
    Calculate the noise level for a given sampling step index.

    Parameters:
    i (int): Index of the current sampling step (0-based index).
    N (int): Total number of sampling steps.
    sigma_min (float): Minimum sigma value for min noise level, default 0.002.
    sigma_max (float): Maximum sigma value for max noise level, default 80.0.
    rho (int): Discretization parameter, default 3 for SD-XL, 7 for EDM2.

    Returns:
    float: Calculated noise level for the given step.
    """
    if i == 0:
        return sigma_max
    if i >= N:
        return 0.0
    sigma_max_p = sigma_max ** (1/rho)
    sigma_min_p = sigma_min ** (1/rho)
    inner_term = sigma_max_p + (i / (N - 1)) * (sigma_min_p - sigma_max_p)
    noise_level = inner_term ** rho

    return noise_level


def find_closest_index(noise_level: float, N: int, sigma_min=0.002, sigma_max=80.0, rho=3, tol=1e-6):
    """
    Given a noise level, find the closest integer index in the range [0, N-1] that corresponds to the noise level.

    Parameters:
    noise_level (float): Target noise level to find the closest index for.
    N (int): Total number of sampling steps.
    sigma_min (float): Minimum sigma value for min noise level, default 0.002.
    sigma_max (float): Maximum sigma value for max noise level, default 80.0.
    rho (int): Discretization parameter, default 3 for SD-XL, 7 for EDM2.

    Returns:
    int: The closest index to the specified noise level.
    """
    # Min/max noise levels for the given range
    if noise_level <= sigma_min:
        return N
    if noise_level >= sigma_max:
        return 0
        #return N - 1
    
    low, high = 0, N - 1
    while low <= high:
        mid = (low + high) // 2
        mid_nl = calculate_noise_level(mid, N)
        if abs(mid_nl - noise_level) < tol:
            return mid
        elif mid_nl < noise_level:
            high = mid - 1
        else:
            low = mid + 1
    
    # If exact match not found, return the index with noise level closest to the target
    return low if abs(calculate_noise_level(low, N) - noise_level) < abs(calculate_noise_level(high, N) - noise_level) else high


### CFG Schedulers


# TODO: Refactor this into something cleaner
def cfg_scheduler(schedule: str, step: int, max_steps: int, w0: float) -> float:
        """
        Constant scheduler for CFG guidance weight.

        Parameters:
        step (int): Current sampling step.
        max_steps (int): Total number of sampling steps.
        w0 (float): Constant value for the guidance weight.

        Returns:
        float: Scheduled guidance weight value.
        """
        match schedule:
                case 'Constant':
                        return constant_schedule(step, max_steps, w0)
                case 'Linear':
                        return linear_schedule(step, max_steps, w0)
                case 'Clamp-Linear (c=4.0)':
                        return clamp_linear_schedule(step, max_steps, w0, 4.0)
                case 'Clamp-Linear (c=2.0)':
                        return clamp_linear_schedule(step, max_steps, w0, 2.0)
                case 'Clamp-Linear (c=1.0)':
                        return clamp_linear_schedule(step, max_steps, w0, 1.0)
                case 'Inverse-Linear':
                        return invlinear_schedule(step, max_steps, w0)
                case 'PCS (s=0.01)':
                        return powered_cosine_schedule(step, max_steps, w0, 0.01)
                case 'PCS (s=0.1)':
                        return powered_cosine_schedule(step, max_steps, w0, 0.1)
                case 'PCS (s=1.0)':
                        return powered_cosine_schedule(step, max_steps, w0, 1.0)
                case 'PCS (s=2.0)':
                        return powered_cosine_schedule(step, max_steps, w0, 2.0)
                case 'PCS (s=4.0)':
                        return powered_cosine_schedule(step, max_steps, w0, 4.0)
                case 'Clamp-Cosine (c=4.0)':
                        return clamp_cosine_schedule(step, max_steps, w0, 4.0)
                case 'Clamp-Cosine (c=2.0)':
                        return clamp_cosine_schedule(step, max_steps, w0, 2.0)
                case 'Clamp-Cosine (c=1.0)':
                        return clamp_cosine_schedule(step, max_steps, w0, 1.0)
                case 'Cosine':
                        return cosine_schedule(step, max_steps, w0)
                case 'Sine':
                        return sine_schedule(step, max_steps, w0)
                case 'V-Shape':
                        return v_shape_schedule(step, max_steps, w0)
                case 'A-Shape':
                        return a_shape_schedule(step, max_steps, w0)
                case 'Interval':
                        return interval_schedule(step, max_steps, w0, 0.25, 5.42)
                case _:
                        logger.error(f"Invalid CFG schedule: {schedule}")
                        return constant_schedule(step, max_steps, w0)


def constant_schedule(step: int, max_steps: int, w0: float):
        """
        Constant scheduler for CFG guidance weight.
        """
        return w0


def linear_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized linear scheduler for CFG guidance weight.
        Such that integral 0-> T ~ w(t) dt  = w*T
        """
        # return w0 * (1 - step / max_steps)
        return w0 * 2 * (1 - step / max_steps)


def clamp_linear_schedule(step: int, max_steps: int, w0: float, c: float):
        """
        Normalized clamp-linear scheduler for CFG guidance weight.
        """
        return max(c, linear_schedule(step, max_steps, w0))


def clamp_cosine_schedule(step: int, max_steps: int, w0: float, c: float):
        """
        Normalized clamp-cosine scheduler for CFG guidance weight.
        """
        return max(c, cosine_schedule(step, max_steps, w0))


def invlinear_schedule(step: int, max_steps: int, w0: float):
        """ 
        Normalized inverse linear scheduler for CFG guidance weight.
        """
        # return w0 * (step / max_steps)
        return w0 * 2 * (step / max_steps)


def powered_cosine_schedule(step: int, max_steps: int, w0: float, s: float):
        """
        Normalized cosine scheduler for CFG guidance weight.
        """
        return w0 * ((1 - math.cos(math.pi * ((max_steps - step) / max_steps)**s))/2.0)


def cosine_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized cosine scheduler for CFG guidance weight.
        """
        return w0 * (1 + math.cos(math.pi * step / max_steps))


def sine_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized sine scheduler for CFG guidance weight.
        """
        return w0 * (math.sin((math.pi * step / max_steps) - (math.pi / 2)) + 1) 


def v_shape_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized V-shape scheduler for CFG guidance weight.
        """
        if step < max_steps / 2:
                return invlinear_schedule(step, max_steps, w0)
        return linear_schedule(step, max_steps, w0)


def a_shape_schedule(step: int, max_steps: int, w0: float):
        """
        Normalized A-shape scheduler for CFG guidance weight.
        """
        if step < max_steps / 2:
                return linear_schedule(step, max_steps, w0)
        return invlinear_schedule(step, max_steps, w0)


def interval_schedule(step: int, max_steps: int, w0: float, low: float, high: float):
        """
        Normalized interval scheduler for CFG guidance weight.
        """
        if low <= step <= high:
                return w0
        return 1.0



# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def pag_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "pag_active"):
                setattr(p, "pag_active", True)
        if 'cfg_interval_' in field and not hasattr(p, "cfg_interval_enable"):
            setattr(p, "cfg_interval_enable", True)
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
def get_mask(attn_modules, r: int=4):
        """ Aggregates the attention across the different layers and heads at the specified resolution. """

        key_corss = f"r{r}_cross"
        key_self = f"r{r}_self"
        curr_r = r

        r_r = 1
        new_ca = 0
        new_fore=0
        a_n=0
        # corresponds to diffusers pipe.unet.config.sample_size
        sample_size = 64
        # get a layer wise mapping
        attention_store_proxy = {"r2_cross": [], "r4_cross": [], "r8_cross": [], "r16_cross": [],
                                 "r2_self": [], "r4_self": [], "r8_self": [], "r16_self": []}
        for module in attn_modules:
                module_type = 'cross' if 'attn2' in module.network_layer_name else 'self'

                attn_map = getattr(module, 'scfg_last_attn_map', None)
                to_q_map = getattr(module, 'scfg_last_to_q_map', None)
                to_k_map = getattr(module, 'scfg_last_to_k_map', None)
                to_v_map = getattr(module, 'scfg_last_to_v_map', None)
                to_out_map = getattr(module, 'scfg_last_to_out_map', None)

                to_q_map = head_to_batch_dim(to_q_map, module.heads)
                to_q_map = average_over_head_dim(to_q_map, module.heads)
                to_q_map = torch.stack([to_q_map[0], to_q_map[0]], dim=0)

                to_k_map = head_to_batch_dim(to_k_map, module.heads)
                to_k_map = average_over_head_dim(to_k_map, module.heads)
                to_k_map = torch.stack([to_k_map[0], to_k_map[0]], dim=0)

                #if getattr(module, 'scfg_last_context_map', None) is not None:
                #        context_map = getattr(module, 'scfg_last_context_map', None)
                #        attn_map = getattr(module, 'scfg_last_attn_map', None)
                #        #attn_map = rearrange(attn_map, 'b n c h w -> b n (h w) c')
                #        module_attn_size = int((attn_map.size(1)) ** (0.5))
                #        r = int(sample_size / module_attn_size)
                #else:
                module_attn_size = int((attn_map.size(1)) ** (0.5))
                r = int(sample_size / module_attn_size)
                #r = int(module_attn_size)
                module_key = f"r{r}_{module_type}"

                batch_size, seq_len, inner_dim = to_out_map.size()
                to_out_map = rearrange(to_out_map, 'b s (h t) -> b h s t', h=module.heads)
                to_out_map = to_out_map.mean(dim=1)
                if not r in [2, 4, 8, 16] or r < 2:
                        continue
                # based on diffusers models/attention.py "get_attention_scores"
                #if module_type == 'self':
                attn_scores = to_q_map @ to_k_map.transpose(-1, -2)
                attn_probs = attn_scores.softmax(dim=-1)
                del attn_scores
                attn_probs = attn_probs.to(to_out_map.dtype)
                #attention_store_proxy[module_key].append(attn_map)
                try:
                        attention_store_proxy[module_key].append(attn_probs)
                        #attention_store_proxy[module_key].append(to_out_map)
                except KeyError:
                        continue

        attention_maps = attention_store_proxy

        #attention_maps = attention_store.get_average_attention()
        while curr_r<=8:
                key_corss = f"r{curr_r}_cross"
                key_self = f"r{curr_r}_self"
                # pdb.set_trace()


                sa = torch.stack(attention_maps[key_self], dim=1)
                ca = torch.stack(attention_maps[key_corss], dim=1)
                attn_num = sa.size(1)
                sa = rearrange(sa, 'b n h w -> (b n) h w')
                ca = rearrange(ca, 'b n h w -> (b n) h w')

                curr = 0 # b hw c=hw
                curr +=sa
                ssgc_sa = curr
                ssgc_n =4
                for _ in range(ssgc_n-1):
                        curr = sa@sa
                        ssgc_sa += curr
                ssgc_sa/=ssgc_n
                sa = ssgc_sa
                ########smoothing ca
                ca = sa@ca # b hw c

                h=w = int(sa.size(1)**(0.5))

                ca = rearrange(ca, 'b (h w) c -> b c h w', h=h )
                if r_r>1:
                        mode =  'bilinear' #'nearest' #
                        ca = F.interpolate(ca, scale_factor=r_r, mode=mode) # b 77 32 32


                #####Gaussian Smoothing
                kernel_size = 3
                sigma = 0.5
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(ca.device)
                channel = ca.size(1)
                ca= rearrange(ca, ' b c h w -> (b c) h w' ).unsqueeze(1)
                ca = F.pad(ca, (1, 1, 1, 1), mode='reflect')
                ca = smoothing(ca.float()).squeeze(1)
                ca = rearrange(ca, ' (b c) h w -> b c h w' , c= channel)
                
                ca_norm = ca/(ca.mean(dim=[2,3], keepdim=True)+1e-8) ### spatial  normlization 
                
                new_ca+=rearrange(ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1) 

                fore_ca = torch.stack([ca[:,0],ca[:,1:].sum(dim=1)], dim=1)
                froe_ca_norm = fore_ca/fore_ca.mean(dim=[2,3], keepdim=True) ### spatial  normlization 
                new_fore += rearrange(froe_ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1)  
                a_n+=attn_num

                curr_r = int(curr_r*2)
                r_r*=2
        
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


        return [ ca_mask, fore_mask]