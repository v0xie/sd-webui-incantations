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

Self-Guidance from arXiv:2412.05827 "Self-Guidance: Boosting Flow and Diffusion Generation on Their Own"
@misc{li2024selfguidanceboostingflowdiffusion,
      title={Self-Guidance: Boosting Flow and Diffusion Generation on Their Own}, 
      author={Tiancheng Li and Weijian Luo and Zhiyang Chen and Liyuan Ma and Guo-Jun Qi},
      year={2024},
      eprint={2412.05827},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05827}, 
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


class SGStateParams:
        def __init__(self):
                self.sg_active: bool = False      # SG guidance scale
                self.sg_sanf: bool = False # saliency-adaptive noise fusion, handled in cfg_combiner
                self.sg_next_sigma: bool = False # use next sigma in noise prediction ( for self-guidance)
                self.sg_disable_perturbation: bool = False # disable perturbation as in SG
                self.sg_scale: int = -1      # SG guidance scale
                self.sg_shift_scale: float = 1.0 # SG shift scale
                self.sg_dynamic_shift_scale: bool = False # dynamic shift scale t/d(t)
                self.step: int = 0
                self.max_sampling_step: int = 0
                self.sg_start_step: int = 0
                self.sg_end_step: int = 150
                self.x_in = None
                self.text_cond = None
                self.image_cond = None
                self.sigmas = None # all the sigmas
                self.sigma = None
                self.text_uncond = None
                self.make_condition_dict = None # callable lambda
                self.crossattn_modules = [] # callable lambda
                self.to_v_modules = []
                self.to_out_modules = []
                self.sg_x_out = None
                self.batch_size = -1      # Batch size
                self.denoiser = None # CFGDenoiser


class SGExtensionScript(UIWrapper):
        def __init__(self):
                pass

        # Extension title in menu UI
        def title(self) -> str:
                return "Self-Guidance"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion('Self-Guidance', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='sg_active')
                        sg_sanf = gr.Checkbox(value=False, default=False, label="Use Saliency-Adaptive Noise Fusion", elem_id='sg_sanf')
                        sg_next_sigma = gr.Checkbox(value=False, default=False, label="Use Next Sigma", elem_id='sg_next_sigma')
                        sg_disable_perturbation = gr.Checkbox(value=False, default=False, label="Disable Perturbation", elem_id='sg_disable_perturbation')
                        sg_dynamic_shift_scale = gr.Checkbox(value=True, default=True, label="Dynamic Shift Scale", elem_id='sg_dynamic_shift_scale')
                        with gr.Row():
                                sg_scale = gr.Slider(value = 0, minimum = 0, maximum = 20.0, step = 0.5, label="SG Scale", elem_id = 'sg_scale', info="")
                                sg_shift_scale = gr.Slider(value = 30.0, minimum = -100, maximum = 100, step = 0.5, label="SG Shift Scale", elem_id = 'sg_shift_scale', info="")
                        with gr.Row():
                                start_step = gr.Slider(value = 0, minimum = 0, maximum = 150, step = 1, label="Start Step", elem_id = 'sg_start_step', info="")
                                end_step = gr.Slider(value = 150, minimum = 0, maximum = 150, step = 1, label="End Step", elem_id = 'sg_end_step', info="")
                active.do_not_save_to_config = True
                sg_sanf.do_not_save_to_config = True
                sg_scale.do_not_save_to_config = True
                sg_shift_scale.do_not_save_to_config = True
                sg_dynamic_shift_scale.do_not_save_to_config = True
                sg_next_sigma.do_not_save_to_config = True
                sg_disable_perturbation.do_not_save_to_config = True
                start_step.do_not_save_to_config = True
                end_step.do_not_save_to_config = True
                self.infotext_fields = [
                        (active, lambda d: gr.Checkbox.update(value='SG Active' in d)),
                        (sg_sanf, lambda d: gr.Checkbox.update(value='SG SANF' in d)),
                        (sg_next_sigma, lambda d: gr.Checkbox.update(value='SG Next Sigma' in d)),
                        (sg_disable_perturbation, lambda d: gr.Checkbox.update(value='SG Disable Perturbation' in d)),
                        (sg_scale, 'SG Scale'),
                        (sg_shift_scale, 'SG Shift Scale'),
                        (sg_dynamic_shift_scale, lambda d: gr.Checkbox.update(value='SG Dynamic Shift Scale' in d)),
                        (start_step, 'SG Start Step'),
                        (end_step, 'SG End Step'),
                ]
                self.paste_field_names = [
                        'sg_active',
                        'sg_sanf',
                        'sg_next_sigma',
                        'sg_disable_perturbation',
                        'sg_scale',
                        'sg_shift_scale',
                        'sg_dynamic_shift_scale',
                        'sg_start_step',
                        'sg_end_step',
                ]
                return [active, sg_scale, start_step, end_step, sg_sanf, sg_next_sigma, sg_disable_perturbation, sg_shift_scale, sg_dynamic_shift_scale]

        def process_batch(self, p: StableDiffusionProcessing, active, sg_scale, start_step, end_step, sg_sanf, sg_next_sigma, sg_disable_perturbation, sg_shift_scale, sg_dynamic_shift_scale, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                active = getattr(p, "sg_active", active)
                sg_sanf = getattr(p, "sg_sanf", sg_sanf)
                sg_next_sigma = getattr(p, "sg_next_sigma", sg_next_sigma)
                sg_disable_perturbation = getattr(p, "sg_disable_perturbation", sg_disable_perturbation)
                sg_scale = getattr(p, "sg_scale", sg_scale)
                sg_shift_scale = getattr(p, "sg_shift_scale", sg_shift_scale)
                sg_dynamic_shift_scale = getattr(p, "sg_dynamic_shift_scale", sg_dynamic_shift_scale)
                start_step = getattr(p, "sg_start_step", start_step)
                end_step = getattr(p, "sg_end_step", end_step)

                if active:
                        p.extra_generation_params.update({
                                "SG Active": active,
                                "SG SANF": sg_sanf,
                                "SG Next Sigma": sg_next_sigma,
                                "SG Disable Perturbation": sg_disable_perturbation,
                                "SG Scale": sg_scale,
                                "SG Shift Scale": sg_shift_scale,
                                "SG Dynamic Shift Scale": sg_dynamic_shift_scale,
                                "SG Start Step": start_step,
                                "SG End Step": end_step,
                        })
                self.create_hook(p, active, sg_scale, start_step, end_step, sg_sanf, sg_next_sigma, sg_disable_perturbation, sg_shift_scale, sg_dynamic_shift_scale)

        def create_hook(self, p: StableDiffusionProcessing, active, sg_scale, start_step, end_step, 
                        sg_sanf, sg_next_sigma, sg_disable_perturbation, sg_shift_scale, sg_dynamic_shift_scale, *args, **kwargs):
                # Create a list of parameters for each concept
                sg_params = SGStateParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['sg_params'] = sg_params
                sg_params.sg_active = active 
                sg_params.sg_sanf = sg_sanf 
                sg_params.sg_next_sigma = sg_next_sigma
                sg_params.sg_disable_perturbation = sg_disable_perturbation
                sg_params.sg_active = active
                sg_params.sg_sanf = sg_sanf
                sg_params.sg_scale = sg_scale
                sg_params.sg_shift_scale = sg_shift_scale
                sg_params.sg_dynamic_shift_scale = sg_dynamic_shift_scale
                sg_params.sg_start_step = start_step
                sg_params.sg_end_step = end_step
                sg_params.batch_size = p.batch_size
                sg_params.denoiser = None

                # Use lambda to call the callback function with the parameters to avoid global variables
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, sg_params)
                cfg_denoised_lambda = lambda callback_params: self.on_cfg_denoised_callback(callback_params, sg_params)
                unhook_lambda = lambda _: self.unhook_callbacks(sg_params)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_cfg_denoised(cfg_denoised_lambda)
                script_callbacks.on_script_unloaded(unhook_lambda)

        def postprocess_batch(self, p, active, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()

                logger.debug('Removed script callbacks')
                active = getattr(p, "sg_active", active)
                if active is False:
                        return

        def remove_all_hooks(self):
                pass

        def unhook_callbacks(self, sg_params: SGStateParams):
                return

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, sg_params: SGStateParams):
                # always unhook
                self.unhook_callbacks(sg_params)

                sg_params.step = params.sampling_step
                # Run SG only if active and within interval
                if not sg_params.sg_active or sg_params.sg_scale <= 0:
                        return
                if not sg_params.sg_start_step <= params.sampling_step <= sg_params.sg_end_step or sg_params.sg_scale <= 0:
                        return

                if isinstance(params.text_cond, dict):
                        text_cond = params.text_cond['crossattn'] # SD XL
                        sg_params.text_cond = {}
                        sg_params.text_uncond = {}
                        for key, value in params.text_cond.items():
                                sg_params.text_cond[key] = value.clone().detach()
                                sg_params.text_uncond[key] = value.clone().detach()
                else:
                        text_cond = params.text_cond # SD 1.5
                        sg_params.text_cond = text_cond.clone().detach()
                        sg_params.text_uncond = text_cond.clone().detach()

                sg_params.x_in = params.x.clone().detach()
                sg_params.sigma = params.sigma.clone().detach()
                sg_params.image_cond = params.image_cond.clone().detach()
                sg_params.denoiser = params.denoiser
                sg_params.make_condition_dict = get_make_condition_dict_fn(params.text_uncond)
                sg_params.sigmas = sg_params.denoiser.sampler.model_wrap.sigmas

        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, sg_params: SGStateParams):
                """ Callback function for the CFGDenoisedParams
                Refer to pg.22 A.2 of the PAG paper for how CFG and PAG combine
                """
                # Run only within interval
                # Run SG only if active and within interval
                # Perturbing last step is weird
                if not sg_params.sg_active or sg_params.sg_scale <= 0:
                        return
                if not sg_params.sg_start_step <= params.sampling_step <= sg_params.sg_end_step or sg_params.sg_scale <= 0:
                        return
                if params.sampling_step >= params.total_sampling_steps-1:
                        return

                # passed from on_cfg_denoiser_callback
                x_in = sg_params.x_in
                tensor = sg_params.text_cond
                uncond = sg_params.text_uncond
                image_cond_in = sg_params.image_cond
                sigma_in = sg_params.sigma 

                if sg_params.sg_next_sigma and sg_params.sg_shift_scale != 0:
                        # calculate next sigma from sigma schedule based on shift scale
                        shift_scale = sg_params.sg_shift_scale
                        current_sigma = sg_params.sigma
                        current_timestep = params.inner_model.sigma_to_t(current_sigma)
                        if sg_params.sg_dynamic_shift_scale:
                                timestep_delta = current_timestep / shift_scale
                        else:
                                timestep_delta = round(sg_params.sg_shift_scale)
                        next_timestep = torch.clamp(current_timestep - timestep_delta, min=0, max=999)
                        next_sigma = params.inner_model.t_to_sigma(next_timestep)
                        sigma_in = next_sigma
                        logger.debug('[SG] Current Sigma: %s Shift amount: %s New Sigma: %s', current_sigma, next_sigma-current_sigma, next_sigma) 

                # "modules/sd_samplers_cfg_denoiser.py:237"
                cond_in = catenate_conds([tensor, uncond])
                make_condition_dict = get_make_condition_dict_fn(uncond)
                conds = make_condition_dict(cond_in, image_cond_in)

                # get the SG guidance (is there a way to optimize this so we don't have to calculate it twice?)
                sg_x_out = params.inner_model(x_in, sigma_in, cond=conds)
                sg_params.sg_x_out = sg_x_out

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[SG] Active", str, sg_apply_override('sg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SG] SANF", str, sg_apply_override('sg_sanf', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SG] Use Next Sigma", str, sg_apply_override('sg_next_sigma', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SG] Disable Perturbation", str, sg_apply_override('sg_disable_perturbation', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SG] SG Scale", float, sg_apply_field("sg_scale")),
                        xyz_grid.AxisOption("[SG] SG Shift Scale", float, sg_apply_field("sg_shift_scale")),
                        xyz_grid.AxisOption("[SG] Dynamic Shift Scale", str, sg_apply_override('sg_dynamic_shift_scale', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[SG] SG Start Step", int, sg_apply_field("sg_start_step")),
                        xyz_grid.AxisOption("[SG] SG End Step", int, sg_apply_field("sg_end_step")),
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
def sg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "sg_active"):
                p.sg_active = True
    return fun


def sg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "sg_active"):
                p.sg_active = True
        setattr(p, field, x)
    return fun
