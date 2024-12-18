import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import torch

from scripts.ui_wrapper import UIWrapper
from modules import shared, script_callbacks, rng
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams
from modules.processing import StableDiffusionProcessing
from modules.sd_samplers_cfg_denoiser import catenate_conds
from scripts.incant_utils import module_hooks

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

incantations_debug = environ.get("INCANTAIONS_DEBUG", False)

"""
Unofficial implementation of TCG from "No Training, No Problem: Rethinking Classifier-Free
Guidance for Diffusion Models" (2024, Sadat et al.)
@misc{sadat2024trainingproblemrethinkingclassifierfree,
      title={No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models}, 
      author={Seyedmorteza Sadat and Manuel Kansy and Otmar Hilliges and Romann M. Weber},
      year={2024},
      eprint={2407.02687},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.02687}, 
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


class TCGStateParams:
        def __init__(self):
                self.tcg_active: bool = False      # TCG guidance scale
                self.tcg_sanf: bool = False # saliency-adaptive noise fusion, handled in cfg_combiner
                self.tcg_scale: float = -1      # TCG guidance scale
                self.tcg_alpha: float = -1      # TCG guidance scale
                self.tcg_std_scale: bool = True
                self.tcg_layers: int = 10
                self.step: int = 0
                self.max_sampling_step: int = 0
                self.guidance_scale: int = 0
                self.tcg_start_step: int = 0
                self.tcg_end_step: int = 150
                self.x_in = None
                self.text_cond = None
                self.image_cond = None
                self.sigma = None
                self.text_uncond = None
                self.make_condition_dict = None # callable lambda
                self.crossattn_modules = [] # callable lambda
                self.time_embed_modules = [] 
                self.to_v_modules = []
                self.to_out_modules = []
                self.tcg_x_out = None
                self.batch_size = -1      # Batch size
                self.denoiser = None # CFGDenoiser


class TCGExtensionScript(UIWrapper):
        def __init__(self):
                pass

        # Extension title in menu UI
        def title(self) -> str:
                return "Time-step Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion(label=self.title(), open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='tcg_active')
                        tcg_sanf = gr.Checkbox(value=False, default=False, label="Use Saliency-Adaptive Noise Fusion", elem_id='tcg_sanf')
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'tcg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'tcg_end_step', info="")
                        with gr.Row():
                                tcg_scale = gr.Slider(value = 0, minimum = 0, maximum = 5.0, step = 0.01, label="TCG Scale", elem_id = 'tcg_scale', info="")
                                tcg_alpha = gr.Slider(value = 0, minimum = 0, maximum = 5.0, step = 0.01, label="TCG Alpha", elem_id = 'tcg_alpha', info="")
                                tcg_max_layer_index = gr.Slider(value = 10, minimum = 1, maximum = 100, step = 1, label="TCG Max Layer Index", elem_id = 'tcg_max_layer_index', info="")
                                tcg_std_scale = gr.Checkbox(value=True, default=True, label="TCG Std Scale", elem_id='tcg_std_scale', info="If enabled, applies TCG with standard deviation scaling")

                                
                active.do_not_save_to_config = True
                tcg_sanf.do_not_save_to_config = True
                tcg_scale.do_not_save_to_config = True
                tcg_alpha.do_not_save_to_config = True
                tcg_max_layer_index.do_not_save_to_config = True
                tcg_std_scale.do_not_save_to_config = True
                start_step.do_not_save_to_config = True
                end_step.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='TCG Active' in d)),
                        (tcg_sanf, lambda d: gr.Checkbox.update(value='TCG SANF' in d)),
                        (start_step, 'TCG Start Step'),
                        (end_step, 'TCG End Step'),
                        (tcg_scale, 'TCG Scale'),
                        (tcg_alpha, 'TCG Alpha'),
                        (tcg_max_layer_index, 'TCG Max Layer Index'),
                        (tcg_std_scale, lambda d: gr.Checkbox.update(value='TCG Std Scale' in d)),
                ]
                self.paste_field_names = [
                        'tcg_active',
                        'tcg_sanf',
                        'tcg_start_step',
                        'tcg_end_step',
                        'tcg_scale',
                        'tcg_alpha',
                        'tcg_max_layer_index',
                        'tcg_std_scale',
                ]
                return [active, start_step, end_step, tcg_sanf, tcg_scale, tcg_alpha, tcg_std_scale, tcg_max_layer_index]

        def process_batch(self, p: StableDiffusionProcessing, active, start_step, end_step, tcg_sanf, tcg_scale, tcg_alpha, tcg_std_scale, tcg_max_layer_index, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "tcg_active", active)
                tcg_sanf = getattr(p, "tcg_sanf", tcg_sanf)
                if active is False:
                        return
                start_step = getattr(p, "tcg_start_step", start_step)
                end_step = getattr(p, "tcg_end_step", end_step)
                tcg_scale = getattr(p, "tcg_scale", tcg_scale)
                tcg_alpha = getattr(p, "tcg_alpha", tcg_alpha)
                tcg_std_scale = getattr(p, "tcg_std_scale", tcg_std_scale)
                tcg_max_layer_index = getattr(p, "tcg_max_layer_index", tcg_max_layer_index)

                if active:
                        p.extra_generation_params.update({
                                "TCG Active": active,
                                "TCG SANF": tcg_sanf,
                                "TCG Scale": tcg_scale,
                                "TCG Start Step": start_step,
                                "TCG End Step": end_step,
                                "TCG Scale": tcg_scale,
                                "TCG Alpha": tcg_alpha,
                                "TCG Std Scale": tcg_std_scale,
                                "TCG Max Layer Index": tcg_max_layer_index,
                        })
                self.create_hook(p, active, start_step, end_step, tcg_sanf, tcg_scale, tcg_alpha, tcg_std_scale, tcg_max_layer_index)

        def create_hook(self, p: StableDiffusionProcessing, active, start_step, end_step, tcg_sanf, tcg_scale, tcg_alpha, tcg_std_scale, tcg_max_layer_index, *args, **kwargs):
                # Create a list of parameters for each concept
                tcg_params = TCGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['tcg_params'] = tcg_params

                tcg_params.tcg_active = active
                tcg_params.tcg_sanf = tcg_sanf
                tcg_params.tcg_scale = tcg_scale
                tcg_params.tcg_start_step = start_step
                tcg_params.tcg_end_step = end_step
                tcg_params.tcg_scale = tcg_scale
                tcg_params.tcg_alpha = tcg_alpha
                tcg_params.tcg_std_scale = tcg_std_scale
                tcg_params.max_sampling_step = p.steps
                tcg_params.guidance_scale = p.cfg_scale
                tcg_params.batch_size = p.batch_size
                tcg_params.denoiser = None

                time_embed_modules = self.get_time_embed_modules(limit=tcg_max_layer_index)
                if len(time_embed_modules) == 0:
                        logger.error("No time embed modules found, cannot apply TCG")
                        return
                tcg_params.time_embed_modules = time_embed_modules

                # Use lambda to call the callback function with the parameters to avoid global variables
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, tcg_params)
                cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(callback_params, tcg_params)
                unhook_lambda = lambda _: self.unhook_callbacks(tcg_params)

                if tcg_params.tcg_scale > 0:
                        self.timestep_hijack_forward(tcg_params.time_embed_modules, tcg_scale, tcg_alpha, tcg_std_scale)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_cfg_denoised(cfg_denoised_lambda)
                script_callbacks.on_script_unloaded(unhook_lambda)

        def postprocess_batch(self, p, *args, **kwargs):
                self.tcg_postprocess_batch(p, *args, **kwargs)

        def tcg_postprocess_batch(self, p, active, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()

                logger.debug('Removed script callbacks')
                active = getattr(p, "tcg_active", active)
                if active is False:
                        return

        def remove_all_hooks(self):
                time_embed_modules = self.get_time_embed_modules(limit=999)
                for module in time_embed_modules:
                        module_hooks.modules_remove_field(module, 'tcg_enable')
                        module_hooks.modules_remove_field(module, 'tcg_scale')
                        module_hooks.modules_remove_field(module, 'tcg_alpha')
                        module_hooks.modules_remove_field(module, 'tcg_timestep')
                        module_hooks.modules_remove_field(module, 'tcg_timestep_max')
                        module_hooks.modules_remove_field(module, 'tcg_timestep_min')
                        module_hooks.modules_remove_field(module, 'tcg_std_scale')
                        module_hooks.remove_module_forward_hook(module, 'tcg_hook')
                        module_hooks.modules_remove_field(module, 'tcg_enable')

        def unhook_callbacks(self, tcg_params: TCGStateParams):
                return

        def timestep_hijack_forward(self, time_embed_modules, tcg_scale, tcg_alpha, tcg_std_scale):
                """ Create hooks in the forward pass of the cross attention modules
                Copies the output of the to_v module to the parent module
                Then applies the TCG perturbation to the output of the cross attention module (multiplication by identity)
                """

                for module in time_embed_modules:
                        module_hooks.modules_add_field(module, 'tcg_enable', False)
                        module_hooks.modules_add_field(module, 'tcg_scale', tcg_scale)
                        module_hooks.modules_add_field(module, 'tcg_alpha', tcg_alpha)
                        module_hooks.modules_add_field(module, 'tcg_std_scale', tcg_std_scale)
                        module_hooks.modules_add_field(module, 'tcg_timestep', 1000)
                        module_hooks.modules_add_field(module, 'tcg_timestep_max', 1000)
                        module_hooks.modules_add_field(module, 'tcg_timestep_min', 400)

                def tcg_hook(module, input, kwargs, output):
                        out_dtype = output.dtype
                        new_output = output.float()
                        if getattr(module, 'tcg_enable', False) is False:
                            return

                        if getattr(module, 'tcg_scale', 0) == 0:
                            return

                        timestep_max = getattr(module, 'tcg_timestep_max', 1000)
                        timestep_min = getattr(module, 'tcg_timestep_min', 0)
                        timestep = getattr(module, 'tcg_timestep', 1000)
                        if not timestep_max >= timestep > timestep_min:
                            return

                        tcg_scale = getattr(module, 'tcg_scale', 0)
                        alpha = getattr(module, 'tcg_alpha', 0)
                        std_scale = getattr(module, 'tcg_std_scale', True)

                        if alpha > 0:
                            noise_scale = tcg_scale * (timestep/1000.0) ** alpha
                        else:
                            noise_scale = tcg_scale
                        # not sure how correct a clamp here is, but it helps if the std is too high
                        if std_scale:
                            noise_scale = noise_scale * torch.clamp(new_output.std(), min=-10, max=10)
                        new_output += rng.randn_like(new_output) * noise_scale
                        if new_output.isnan().any():
                            logger.error(f"NaN in TCG output")
                            return output
                        return new_output.to(out_dtype)

                # Create hooks 
                for module in time_embed_modules:
                        module_hooks.module_add_forward_hook(module, tcg_hook, hook_type="forward", with_kwargs=True)

        def get_time_embed_modules(self, limit=10):
                try:
                        m = shared.sd_model
                        nlm = m.network_layer_mapping
                        time_embed_modules = [m for m in nlm.values() if 'timestep' in m.__class__.__name__.lower()]
                        limit = min(limit, len(time_embed_modules)-1) # pertubring the very last layer is not useful
                        time_embed_modules = time_embed_modules[:limit]
                        return time_embed_modules
                except AttributeError:
                        logger.exception("AttributeError in get_timestep_modules", stack_info=True)
                        return []
                except Exception:
                        logger.exception("Exception in get_timestep_modules", stack_info=True)
                        return []

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, tcg_params: TCGStateParams):
                # always unhook
                self.unhook_callbacks(tcg_params)

                tcg_params.step = params.sampling_step

                # Run TCG only if active and within interval
                if not tcg_params.tcg_active or tcg_params.tcg_scale <= 0:
                        return
                if not tcg_params.tcg_start_step <= params.sampling_step <= tcg_params.tcg_end_step or tcg_params.tcg_scale <= 0:
                        return

                if isinstance(params.text_cond, dict):
                        text_cond = params.text_cond['crossattn'] # SD XL
                        tcg_params.text_cond = {}
                        tcg_params.text_uncond = {}
                        for key, value in params.text_cond.items():
                                tcg_params.text_cond[key] = value.clone().detach()
                                tcg_params.text_uncond[key] = value.clone().detach()
                else:
                        text_cond = params.text_cond # SD 1.5
                        tcg_params.text_cond = text_cond.clone().detach()
                        tcg_params.text_uncond = text_cond.clone().detach()

                tcg_params.x_in = params.x.clone().detach()
                tcg_params.sigma = params.sigma.clone().detach()
                tcg_params.image_cond = params.image_cond.clone().detach()
                tcg_params.denoiser = params.denoiser
                tcg_params.make_condition_dict = get_make_condition_dict_fn(params.text_uncond)

        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, tcg_params: TCGStateParams):
                """ Callback function for the CFGDenoisedParams
                Refer to pg.22 A.2 of the TCG paper for how CFG and TCG combine
                """
                # Run only within interval
                # Run TCG only if active and within interval
                if not tcg_params.tcg_active or tcg_params.tcg_scale <= 0:
                        return
                if not tcg_params.tcg_start_step <= params.sampling_step <= tcg_params.tcg_end_step or tcg_params.tcg_scale <= 0:
                        return

                # passed from on_cfg_denoiser_callback
                x_in = tcg_params.x_in
                tensor = tcg_params.text_cond
                uncond = tcg_params.text_uncond
                image_cond_in = tcg_params.image_cond
                sigma_in = tcg_params.sigma

                # "modules/sd_samplers_cfg_denoiser.py:237"
                cond_in = catenate_conds([tensor, uncond])
                make_condition_dict = get_make_condition_dict_fn(uncond)
                conds = make_condition_dict(cond_in, image_cond_in)

                # set tcg_enable to True for the hooked time embed modules
                # set timestep to a rough estimate
                timestep = max(1000 - (params.sampling_step / params.total_sampling_steps * 1000), 0)
                for module in tcg_params.time_embed_modules:
                        module.tcg_enable = True
                        module.tcg_timestep = timestep

                # get the TCG guidance (is there a way to optimize this so we don't have to calculate it twice?)
                tcg_x_out = params.inner_model(x_in, sigma_in, cond=conds)
                tcg_params.tcg_x_out = tcg_x_out

                # set tcg_enable to False
                for module in tcg_params.time_embed_modules:
                        module.tcg_enable = False

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[TCG] Active", str, tcg_apply_override('tcg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[TCG] TCG Start Step", int, tcg_apply_field("tcg_start_step")),
                        xyz_grid.AxisOption("[TCG] TCG End Step", int, tcg_apply_field("tcg_end_step")),
                        xyz_grid.AxisOption("[TCG] SANF", str, tcg_apply_override('tcg_sanf', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[TCG] Scale", float, tcg_apply_field("tcg_scale")),
                        xyz_grid.AxisOption("[TCG] Alpha", float, tcg_apply_field("tcg_alpha")),
                        xyz_grid.AxisOption("[TCG] Std Scale", str, tcg_apply_override('tcg_std_scale', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[TCG] Max Layer Index", int, tcg_apply_field("tcg_max_layer_index")),
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
def tcg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "tcg_active"):
                setattr(p, "tcg_active", True)
        if 'tcg_' in field and not hasattr(p, "tcg_active"):
                p.tcg_active = True
    return fun


def tcg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "tcg_active"):
                setattr(p, "tcg_active", True)
        if 'tcg_' in field and not hasattr(p, "tcg_active"):
                p.tcg_active = True
        setattr(p, field, x)
    return fun
