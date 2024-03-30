import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import scipy.stats as stats

from scripts.ui_wrapper import UIWrapper, arg
from modules import script_callbacks
from modules.hypernetworks import hypernetwork
#import modules.sd_hijack_optimizations
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams 
from modules.prompt_parser import reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import pad_cond
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

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""

handles = []

class PAGStateParams:
        def __init__(self):
                self.x_in = None
                self.text_cond = None
                self.image_cond = None
                self.sigma = None
                self.text_uncond = None
                self.make_condition_dict = None # callable lambda
                self.crossattn_modules = [] # callable lambda
                self.to_v_modules = []
                self.to_out_modules = []
                self.guidance_scale: int = -1 # CFG
                self.pag_scale: int = -1      # PAG guidance scale
                self.pag_x_out: None
                self.batch_size : int = -1      # Batch size
                self.denoiser: None # CFGDenoiser

                self.attnreg: bool = False
                self.ema_smoothing_factor: float = 2.0
                self.step_end : int = 25
                self.tokens: str = "" # [0, 20]
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
                        with gr.Row():
                                pag_scale = gr.Slider(value = 1.0, minimum = 0, maximum = 20.0, step = 0.5, label="PAG Scale", elem_id = 'pag_scale', info="")
                        step_end = gr.Slider(value=25, minimum=0, maximum=150, default=1, step=1, label="Step End", elem_id='pag_step_end')
                        with gr.Row():
                                tokens = gr.Textbox(visible=False, value="", label="Tokens", elem_id='pag_tokens', info="Comma separated list of tokens to condition on")
                        with gr.Row():
                                correction_threshold = gr.Slider(visible=False, value = 0.0, minimum = 0., maximum = 1.0, step = 0.001, label="CbS Score Threshold", elem_id = 'pag_correction_threshold', info="Filter dimensions with similarity below this threshold")
                                correction_strength = gr.Slider(visible=False, value = 0.0, minimum = 0.0, maximum = 0.999, step = 0.01, label="CbS Correction Strength", elem_id = 'pag_correction_strength', info="The strength of the correction, default 0.1")
                        with gr.Row():
                                attnreg = gr.Checkbox(visible=False, value=False, default=False, label="Use Attention Regulation", elem_id='pag_use_attnreg')
                                ctnms_alpha = gr.Slider(visible=False, value = 0.1, minimum = 0.0, maximum = 1.0, step = 0.01, label="Alpha for Cross-Token Non-Maximum Suppression", elem_id = 'pag_ctnms_alpha', info="Contribution of the suppressed attention map, default 0.1")
                                ema_factor = gr.Slider(visible=False, value=0.0, minimum=0.0, maximum=4.0, default=2.0, label="EMA Smoothing Factor", elem_id='pag_ema_factor')
                active.do_not_save_to_config = True
                pag_scale.do_not_save_to_config = True
                attnreg.do_not_save_to_config = True
                ema_factor.do_not_save_to_config = True
                step_end.do_not_save_to_config = True
                ctnms_alpha.do_not_save_to_config = True
                correction_threshold.do_not_save_to_config = True
                correction_strength.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='PAG Active' in d)),
                        (pag_scale, 'PAG Scale'),
                        #(attnreg, lambda d: gr.Checkbox.update(value='PAG AttnReg' in d)),
                        # (step_end, 'PAG Step End'),
                        # (ctnms_alpha, 'PAG CTNMS Alpha'),
                        # (correction_threshold, 'PAG CbS Score Threshold'),
                        # (correction_strength, 'PAG CbS Correction Strength'),
                        # (ema_factor, 'PAG CTNMS EMA Smoothing Factor'),
                ]
                self.paste_field_names = [
                        'pag_active',
                        'pag_scale',
                        #'pag_attnreg',
                        #'pag_ctnms_alpha',
                        #'pag_correction_threshold',
                        #'pag_correction_strength'
                        #'pag_ema_factor',
                        #'pag_step_end'
                ]
                return [active, attnreg, pag_scale, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.pag_process_batch(p, *args, **kwargs)

        def pag_process_batch(self, p: StableDiffusionProcessing, active, attnreg, pag_scale, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end, *args, **kwargs):
                #self.unhook_callbacks()

                active = getattr(p, "pag_active", active)
                use_attnreg = getattr(p, "pag_attnreg", attnreg)
                ema_factor = getattr(p, "pag_ema_factor", ema_factor)
                step_end = getattr(p, "pag_step_end", step_end)
                if active is False:
                        return
                pag_scale = getattr(p, "pag_scale", pag_scale)
                ctnms_alpha = getattr(p, "pag_ctnms_alpha", ctnms_alpha)
                correction_threshold = getattr(p, "pag_correction_threshold", correction_threshold)
                correction_strength = getattr(p, "pag_correction_strength", correction_strength)
                tokens = getattr(p, "pag_tokens", tokens)
                p.extra_generation_params.update({
                        "PAG Active": active,
                        "PAG Scale": pag_scale,
                        #"PAG AttnReg": attnreg,
                        #"PAG Tokens": tokens,
                        # "PAG CbS Score Threshold": correction_threshold,
                        # "PAG CbS Correction Strength": correction_strength,
                        # "PAG CTNMS Alpha": ctnms_alpha,
                        # "PAG Step End": step_end,
                        # "PAG EMA Smoothing Factor": ema_factor,
                })

                self.create_hook(p, active, attnreg, pag_scale, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end, p.width, p.height)

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

        def create_hook(self, p: StableDiffusionProcessing, active, attnreg, pag_scale, ctnms_alpha, correction_threshold, correction_strength, tokens, ema_factor, step_end, width, height, *args, **kwargs):
                # Create a list of parameters for each concept
                pag_params = []

                #for _, strength in concept_conds:
                params = PAGStateParams()
                params.guidance_scale = p.cfg_scale
                params.pag_scale = pag_scale
                params.batch_size = p.batch_size

                params.attnreg = attnreg 
                params.ema_smoothing_factor = ema_factor 
                params.step_end = step_end 
                params.ctnms_alpha = ctnms_alpha
                params.correction_threshold = correction_threshold
                params.correction_strength = correction_strength
                params.strength = 1.0
                params.width = width
                params.height = height 
                params.dims = [width, height]

                # Get all the qv modules
                cross_attn_modules = self.get_cross_attn_modules()
                params.crossattn_modules = [m for m in cross_attn_modules if 'CrossAttention' in m.__class__.__name__]
                pag_params.append(params)

                setattr(p, "pag_params", pag_params)

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, pag_params)
                z = lambda params: self.on_cfg_denoised_callback(params, pag_params)
                after_cfg_callback = lambda params: self.cfg_after_cfg_callback(params, pag_params)
                un = lambda pag_params: self.unhook_callbacks(None)

                self.ready_hijack_forward(params.crossattn_modules, pag_scale, width, height, ema_factor, step_end)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_cfg_denoised(z)
                script_callbacks.cfg_after_cfg_callback(after_cfg_callback)

                script_callbacks.on_script_unloaded(un)

        def postprocess_batch(self, p, *args, **kwargs):
                self.pag_postprocess_batch(p, *args, **kwargs)

        def pag_postprocess_batch(self, p, active, *args, **kwargs):
                sampler = getattr(p, "sampler", None)
                if sampler is None:
                        return
                denoiser = getattr(sampler, "model_wrap_cfg", None)
                if denoiser is None:
                        return

                #pag_params = getattr(p, "pag_params", None)
                self.unhook_callbacks(denoiser)
                active = getattr(p, "pag_active", active)
                if active is False:
                        return

        def unhook_callbacks(self, denoiser=None):
                global handles
                logger.debug('Unhooked callbacks')
                cross_attn_modules = self.get_cross_attn_modules()
                for module in cross_attn_modules:
                        to_v = getattr(module, 'to_v', None)
                        to_out = getattr(module, 'to_out', None)
                        self.remove_field_cross_attn_modules(module, 'pag_enable')
                        self.remove_field_cross_attn_modules(module, 'pag_scale')
                        self.remove_field_cross_attn_modules(module, 'pag_last_to_v')
                        self.remove_field_cross_attn_modules(to_v, 'pag_parent_module')
                        self.remove_field_cross_attn_modules(to_out, 'pag_parent_module')
                        _remove_all_forward_hooks(module, 'pag_pre_hook')
                        _remove_all_forward_hooks(to_v, 'to_v_pre_hook')
                if denoiser:
                        self.unpatch_combine_denoised(denoiser)
                script_callbacks.remove_current_script_callbacks()

        def ready_hijack_forward(self, crossattn_modules, pag_scale, width, height, ema_factor, step_end):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                Then applies the PAG perturbation to the output of the cross attention module (multiplication by identity)
                """

                # add field for last_to_v
                for module in crossattn_modules:
                        to_v = getattr(module, 'to_v', None)
                        to_out = getattr(module, 'to_out', None)
                        self.add_field_cross_attn_modules(module, 'pag_enable', False)
                        self.add_field_cross_attn_modules(module, 'pag_scale', torch.tensor([pag_scale], dtype=torch.float16, device=shared.device))
                        self.add_field_cross_attn_modules(module, 'pag_last_to_v', None)
                        self.add_field_cross_attn_modules(to_v, 'pag_parent_module', [module])
                        self.add_field_cross_attn_modules(to_out, 'pag_parent_module', [module])

                def to_v_pre_hook(module, input, kwargs, output):
                        """ Copy the output of the to_v module to the parent module """
                        parent_module = getattr(module, 'pag_parent_module', None)
                        # copy the output of the to_v module to the parent module
                        setattr(parent_module[0], 'pag_last_to_v', output.detach().clone())

                def pag_pre_hook(module, input, kwargs, output):
                        if hasattr(module, 'pag_enable') and getattr(module, 'pag_enable', False) is False:
                                return
                        if not hasattr(module, 'pag_last_to_v'):
                                # oops we forgot to unhook
                                return
                        #print(f"Pag enabled")

                        batch_size, seq_len, inner_dim = output.shape
                        identity = torch.eye(seq_len).expand(batch_size, -1, -1).to(output.device)

                        # get the last to_v output
                        last_to_v = getattr(module, 'pag_last_to_v', None)

                        if last_to_v is not None:    
                                new_output = torch.einsum('bij,bjk->bik', identity, last_to_v)
                                return new_output
                        else:
                                # this is bad
                                return output

                # Create hooks 
                for module in crossattn_modules:
                        handle_parent = module.register_forward_hook(pag_pre_hook, with_kwargs=True)
                        to_v = getattr(module, 'to_v', None)
                        handle_to_v = to_v.register_forward_hook(to_v_pre_hook, with_kwargs=True)


        def get_middle_block_modules(self):
                """ Get all attention modules from the middle block 
                Refere to page 22 of the PAG paper, Appendix A.2
                
                """
                m = shared.sd_model
                nlm = m.network_layer_mapping
                middle_block_modules = [m for m in nlm.values() if 'middle_block_1_transformer_blocks_0_attn1' in m.network_layer_name and 'CrossAttention' in m.__class__.__name__]
                return middle_block_modules

        def get_cross_attn_modules(self):
                """ Get all croos attention modules """
                return self.get_middle_block_modules()
                # m = shared.sd_model
                # nlm = m.network_layer_mapping
                # cross_attn_modules = [m for m in nlm.values() if 'Attention' in m.__class__.__name__]
                # cross_attn_modules = [m for m in nlm.values() if 'attn' in m.__class__.__name__]
                # return cross_attn_modules

        def add_field_cross_attn_modules(self, module, field, value):
                """ Add a field to a module if it doesn't exist """
                if not hasattr(module, field):
                        setattr(module, field, value)
        
        def remove_field_cross_attn_modules(self, module, field):
                """ Remove a field from a module if it exists """
                if hasattr(module, field):
                        delattr(module, field)

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: list[PAGStateParams]):
                # self.unpatch_combine_denoised(params.denoiser)

                if isinstance(params.text_cond, dict):
                        text_cond = params.text_cond['crossattn'] # SD XL
                else:
                        text_cond = params.text_cond # SD 1.5

                        pag_params[0].x_in = params.x.clone().detach()
                        pag_params[0].text_cond = params.text_cond.clone().detach()
                        pag_params[0].sigma = params.sigma.clone().detach()
                        pag_params[0].image_cond = params.image_cond.clone().detach()
                        pag_params[0].text_uncond = params.text_cond.clone().detach()
                        pag_params[0].denoiser = params.denoiser

                # assign callable lambda to make_condition_dict
                if shared.sd_model.model.conditioning_key == "crossattn-adm":
                        pag_params[0].make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": [c_crossattn], "c_adm": c_adm}
                else:
                        if isinstance(pag_params[0].text_uncond, dict):
                                pag_params[0].make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
                        else:
                                pag_params[0].make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}


        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, pag_params: list[PAGStateParams]):
                """ Callback function for the CFGDenoisedParams 
                This is where we combine CFG and PAG
                Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine

                s: PAG guidance scale, w: CFG guidance scale


                Normal CFG calculation (in combined_denoised() in modules/sd_samplers_cfg_denoiser.py)
                        CFG = w(ϵθ (xt, c) − ϵθ (xt, ϕ))
                
                PAG guidance calculation (in on_cfg_denoiser_callback)
                        PAG = s(ϵθ (xt, c) − ˆϵθ (xt, c))

                To work around the issue of CFG guidance being applied to the PAG guidance, 
                we can use the following formulation:

                ϵθ (xt, c) = x_out[0]
                ϵθ (xt, ϕ) = x_out[1]
                ˆϵθ (xt, c) = pag_x_out[1]

                ~ϵθ (xt, c) = ϵθ (xt, c) + CFG + PAG
                ~ϵθ (xt, c) = ϵθ (xt, c) + w(ϵθ (xt, c) − ϵθ (xt, ϕ)) + s(ϵθ (xt, c) − ˆϵθ (xt, c))  
                ~ϵθ (xt, c) = ϵθ (xt, c) + w(CFG) + s(PAG)  
                ~ϵθ (xt, c) = ϵθ (xt, c) + w(CFG + s/w(PAG))
                
                """
                # original x_out
                x_out = params.x

                # passed from on_cfg_denoiser_callback
                x_in = pag_params[0].x_in
                tensor = pag_params[0].text_cond
                uncond = pag_params[0].text_uncond
                image_cond_in = pag_params[0].image_cond
                sigma_in = pag_params[0].sigma
                
                # concatenate the conditions 
                # "modules/sd_samplers_cfg_denoiser.py:237"
                cond_in = catenate_conds([tensor, uncond])
                make_condition_dict = pag_params[0].make_condition_dict
                conds = make_condition_dict(cond_in, image_cond_in)
                #setattr(conds, "pag_active", True)
                
                # set pag_enable to True
                for module in pag_params[0].crossattn_modules:
                        setattr(module, 'pag_enable', True)

                # get the PAG guidance
                pag_x_out = params.inner_model(x_in, sigma_in, cond=conds)

                # combine CFG and PAG
                pag_scale = pag_params[0].pag_scale
                cfg_scale = pag_params[0].guidance_scale

                # update pag_x_out
                pag_params[0].pag_x_out = pag_x_out

                # pag_x_out[-uncond.shape[0]:] = x_out[-uncond.shape[0]:] + (pag_scale/cfg_scale)*(x_out[-uncond.shape[0]:] - pag_x_out[-uncond.shape[0]:])

                # update x_out
                #x_out[-uncond.shape[0]:] = pag_x_out[-uncond.shape[0]:]
                params.x = x_out

                # set pag_enable to False
                for module in pag_params[0].crossattn_modules:
                        setattr(module, 'pag_enable', False)
                
                # patch combine_denoised
                self.patch_combine_denoised(pag_params)
        
        def cfg_after_cfg_callback(self, params, pag_params):
                pass
                #self.unpatch_combine_denoised(pag_params)

        def patch_combine_denoised(self, pag_params):
                # pag_scale = pag_params[0].pag_scale
                # pag_x_out = pag_params[0].pag_x_out
                if hasattr(pag_params[0].denoiser, 'original_combine_denoised'):
                        self.unpatch_combine_denoised(pag_params[0].denoiser)
                        #pag_params[0].pag_scale = pag_scale
                        #pag_params[0].pag_x_out = pag_x_out
                #if not hasattr(pag_params[0].denoiser, 'pag_wrapped'):
                original_func = pag_params[0].denoiser.combine_denoised
                        #setattr(pag_params[0].denoiser, 'pag_wrapped', True)
                setattr(pag_params[0].denoiser, 'original_combine_denoised', original_func)
                setattr(pag_params[0].denoiser, 'combine_denoised', wrap_combined_denoised_with_extra_cond(pag_params))
                print("Patched combine_denoised")
        
        def unpatch_combine_denoised(self, denoiser):
                if hasattr(denoiser, 'original_combine_denoised'):
                        original_func = denoiser.original_combine_denoised
                        setattr(denoiser, 'combine_denoised', original_func)
                else:
                       setattr(denoiser, 'combine_denoised', CFGDenoiser.combine_denoised)
                print("Unpatched combine_denoised")

        # def on_cfg_denoised_callback(self, params, pag_params: list[PAGStateParams]):
        #         denoiser = pag_params[0].denoiser

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[PAG] Active", str, pag_apply_override('pag_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] PAG Scale", float, pag_apply_field("pag_scale")),
                        #xyz_grid.AxisOption("[PAG] ctnms_alpha", float, pag_apply_field("pag_ctnms_alpha")),
                        #xyz_grid.AxisOption("[PAG] Step End", float, pag_apply_field("pag_step_end")),
                        #xyz_grid.AxisOption("[PAG] Correction Threshold", float, pag_apply_field("pag_correction_threshold")),
                        #xyz_grid.AxisOption("[PAG] Correction Strength", float, pag_apply_field("pag_correction_strength")),
                        #xyz_grid.AxisOption("[PAG] CTNMS EMA Smoothing Factor", float, pag_apply_field("pag_ema_factor")),
                }
                return extra_axis_options


def wrap_combined_denoised_with_extra_cond(pag_params):
        def combine_denoised_wrapped(x_out, conds_list, uncond, cond_scale):
                denoised_uncond = x_out[-uncond.shape[0]:]
                denoised_uncond_pag = pag_params[0].pag_x_out[-uncond.shape[0]:]
                denoised = torch.clone(denoised_uncond)

                for i, conds in enumerate(conds_list):
                        for cond_index, weight in conds:
                                denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cond_scale)
                                denoised[i] += (x_out[cond_index] - denoised_uncond_pag[i]) * (weight * pag_params[0].pag_scale)
                return denoised
        return combine_denoised_wrapped 

# class PAGDenoiser(sd_samplers_cfg_denoiser.CFGDenoiser):
#         @property
#         def inner_model(self):
#                 if self.model_wrap is None:
#                         denoiser = k_diffusion.external.CompVisVDenoiser if shared.sd_model.parameterization == "v" else k_diffusion.external.CompVisDenoiser
#                         self.model_wrap = denoiser(shared.sd_model, quantize=shared.opts.enable_quantization)
# 
#                 return self.model_wrap

# Hijacks

def catenate_conds(conds):
    if not isinstance(conds[0], dict):
        return torch.cat(conds)

    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


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