import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import torch

from scripts.ui_wrapper import UIWrapper
from modules import shared, script_callbacks
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams
from modules.processing import StableDiffusionProcessing
from modules.sd_samplers_cfg_denoiser import catenate_conds
from scripts.incant_utils import module_hooks

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

Saliency-adaptive noise fusion from arXiv:2311.10329 "High-fidelity Person-centric Subject-to-Image Synthesis"
@misc{wang2024highfidelity,
      title={High-fidelity Person-centric Subject-to-Image Synthesis}, 
      author={Yibin Wang and Weizhong Zhang and Jianwei Zheng and Cheng Jin},
      year={2024},
      eprint={2311.10329},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


class PAGStateParams:
        def __init__(self):
                self.pag_active: bool = False      # PAG guidance scale
                self.pag_sanf: bool = False # saliency-adaptive noise fusion, handled in cfg_combiner
                self.pag_scale: int = -1      # PAG guidance scale
                self.step: int = 0
                self.pag_start_step: int = 0
                self.pag_end_step: int = 150 
                self.x_in = None
                self.text_cond = None
                self.image_cond = None
                self.sigma = None
                self.text_uncond = None
                self.make_condition_dict = None # callable lambda
                self.crossattn_modules = [] # callable lambda
                self.to_v_modules = []
                self.to_out_modules = []
                self.pag_x_out = None
                self.batch_size = -1      # Batch size
                self.denoiser = None # CFGDenoiser


class PAGExtensionScript(UIWrapper):
        def __init__(self):
                pass

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
                        pag_sanf = gr.Checkbox(value=False, default=False, label="Use Saliency-Adaptive Noise Fusion", elem_id='pag_sanf')
                        with gr.Row():
                                pag_scale = gr.Slider(value = 0, minimum = 0, maximum = 20.0, step = 0.5, label="PAG Scale", elem_id = 'pag_scale', info="")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'pag_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'pag_end_step', info="")
                active.do_not_save_to_config = True
                pag_sanf.do_not_save_to_config = True
                pag_scale.do_not_save_to_config = True
                start_step.do_not_save_to_config = True
                end_step.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='PAG Active' in d)),
                        (pag_sanf, lambda d: gr.Checkbox.update(value='PAG SANF' in d)),
                        (pag_scale, 'PAG Scale'),
                        (start_step, 'PAG Start Step'),
                        (end_step, 'PAG End Step'),
                ]
                self.paste_field_names = [
                        'pag_active',
                        'pag_sanf',
                        'pag_scale',
                        'pag_start_step',
                        'pag_end_step',
                ]
                return [active, pag_scale, start_step, end_step, pag_sanf]

        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
               self.pag_process_batch(p, *args, **kwargs)

        def pag_process_batch(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, pag_sanf, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "pag_active", active)
                pag_sanf = getattr(p, "pag_sanf", pag_sanf)
                if active is False:
                        return
                pag_scale = getattr(p, "pag_scale", pag_scale)
                start_step = getattr(p, "pag_start_step", start_step)
                end_step = getattr(p, "pag_end_step", end_step)

                if active:
                        p.extra_generation_params.update({
                                "PAG Active": active,
                                "PAG SANF": pag_sanf,
                                "PAG Scale": pag_scale,
                                "PAG Start Step": start_step,
                                "PAG End Step": end_step,
                        })
                self.create_hook(p, active, pag_scale, start_step, end_step, pag_sanf)

        def create_hook(self, p: StableDiffusionProcessing, active, pag_scale, start_step, end_step, pag_sanf, *args, **kwargs):
                # Create a list of parameters for each concept
                pag_params = PAGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['pag_params'] = pag_params
                
                pag_params.pag_active = active 
                pag_params.pag_sanf = pag_sanf 
                pag_params.pag_scale = pag_scale
                pag_params.pag_start_step = start_step
                pag_params.pag_end_step = end_step
                pag_params.batch_size = p.batch_size
                pag_params.denoiser = None

                # Get all the qv modules
                cross_attn_modules = self.get_cross_attn_modules()
                if len(cross_attn_modules) == 0:
                        logger.error("No cross attention modules found, cannot proceed with PAG")
                        return
                pag_params.crossattn_modules = [m for m in cross_attn_modules if 'CrossAttention' in m.__class__.__name__]

                # Use lambda to call the callback function with the parameters to avoid global variables
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, pag_params)
                cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(callback_params, pag_params)
                unhook_lambda = lambda _: self.unhook_callbacks(pag_params)

                if pag_params.pag_active:
                        self.ready_hijack_forward(pag_params.crossattn_modules, pag_scale)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_cfg_denoised(cfg_denoised_lambda)
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
                        module_hooks.modules_remove_field(module, 'pag_enable')
                        module_hooks.modules_remove_field(module, 'pag_last_to_v')
                        module_hooks.modules_remove_field(module.to_v, 'pag_parent_module')
                        module_hooks.remove_module_forward_hook(module, 'pag_pre_hook')
                        module_hooks.remove_module_forward_hook(module.to_v, 'to_v_pre_hook')

        def unhook_callbacks(self, pag_params: PAGStateParams):
                return

        def ready_hijack_forward(self, crossattn_modules, pag_scale):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                Then applies the PAG perturbation to the output of the cross attention module (multiplication by identity)
                """

                # add field for last_to_v
                for module in crossattn_modules:
                        module_hooks.modules_add_field(module, 'pag_enable', False)
                        module_hooks.modules_add_field(module, 'pag_last_to_v', None)
                        module_hooks.modules_add_field(module.to_v, 'pag_parent_module', [module])

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

                        # get the last to_v output and save it
                        last_to_v = getattr(module, 'pag_last_to_v', None)

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
                        module_hooks.module_add_forward_hook(module, pag_pre_hook, hook_type="forward", with_kwargs=True)
                        module_hooks.module_add_forward_hook(module.to_v, to_v_pre_hook, hook_type="forward", with_kwargs=True)

        def get_middle_block_modules(self):
                """ Get all attention modules from the middle block 
                Refere to page 22 of the PAG paper, Appendix A.2
                
                """
                try:
                        #m = shared.sd_model
                        #nlm = m.network_layer_mapping
                        #middle_block_modules = [m for m in nlm.values() if 'middle_block_1_transformer_blocks_0_attn1' in m.network_layer_name and 'CrossAttention' in m.__class__.__name__]
                        middle_block_modules = module_hooks.get_modules(
                               network_layer_name_filter='middle_block_1_transformer_blocks_0_attn1',
                               module_name_filter = 'CrossAttention'
                        )
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

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, pag_params: PAGStateParams):
                # always unhook
                self.unhook_callbacks(pag_params)

                pag_params.step = params.sampling_step

                # Run PAG only if active and within interval
                if not pag_params.pag_active or pag_params.pag_scale <= 0:
                        return
                if not pag_params.pag_start_step <= params.sampling_step <= pag_params.pag_end_step or pag_params.pag_scale <= 0:
                        return

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

        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, pag_params: PAGStateParams):
                """ Callback function for the CFGDenoisedParams 
                Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine
                
                """
                # Run only within interval
                # Run PAG only if active and within interval
                if not pag_params.pag_active or pag_params.pag_scale <= 0:
                        return
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
                pag_params.pag_x_out = pag_x_out

                # set pag_enable to False
                for module in pag_params.crossattn_modules:
                        setattr(module, 'pag_enable', False)
        
        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[PAG] Active", str, pag_apply_override('pag_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] SANF", str, pag_apply_override('pag_sanf', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[PAG] PAG Scale", float, pag_apply_field("pag_scale")),
                        xyz_grid.AxisOption("[PAG] PAG Start Step", int, pag_apply_field("pag_start_step")),
                        xyz_grid.AxisOption("[PAG] PAG End Step", int, pag_apply_field("pag_end_step")),
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
def pag_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "pag_active"):
                setattr(p, "pag_active", True)
    return fun


def pag_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "pag_active"):
                setattr(p, "pag_active", True)
        setattr(p, field, x)
    return fun
