import logging
from os import environ
import modules.scripts as scripts
import gradio as gr

from scripts.ui_wrapper import UIWrapper
from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams
from modules.processing import StableDiffusionProcessing

import math


logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

incantations_debug = environ.get("INCANTAIONS_DEBUG", False)

"""
An unofficial implementation of CFG schedulers from "Analysis of Classifier-Free Guidance Weight Schedulers"

@misc{wang2024analysis,
      title={Analysis of Classifier-Free Guidance Weight Schedulers},
      author={Xi Wang and Nicolas Dufour and Nefeli Andreou and Marie-Paule Cani and Victoria Fernandez Abrevaya and David Picard and Vicky Kalogeiton},
      year={2024},
      eprint={2404.13040},
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

EP-CFG from "EP-CFG: Energy-Preserving Classifier-Free Guidance"
@misc{zhang2024epcfgenergypreservingclassifierfreeguidance,
      title={EP-CFG: Energy-Preserving Classifier-Free Guidance}, 
      author={Kai Zhang and Fujun Luan and Sai Bi and Jianming Zhang},
      year={2024},
      eprint={2412.09966},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.09966}, 
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


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


class CFGSchedulerParams:
        def __init__(self):
                self.cfg_interval_enable: bool = False
                self.cfg_interval_schedule: str = 'Constant'
                self.cfg_interval_low: float = 0
                self.cfg_interval_high: float = 50.0
                self.cfg_interval_scheduled_value: float = 7.0
                self.step : int = 0
                self.max_sampling_step : int = 1
                self.guidance_scale: int = -1 # CFG
                self.current_noise_level: float = 100.0
                self.ep_cfg_enable: bool = False
                self.ep_cfg_min: float = 0.45 # noise thresholds
                self.ep_cfg_max: float = 0.55


class CFGSchedulerExtensionScript(UIWrapper):
        def __init__(self):
                self.cached_c = [None, None]
                self.handles = []

        # Extension title in menu UI
        def title(self) -> str:
                return "CFG Scheduler"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img) -> list:
                with gr.Accordion(label=self.title(), open=False):
                        with gr.Row():
                                ep_cfg_enable = gr.Checkbox(value=False, label="EP-CFG Enable", elem_id='ep_cfg_interval_enable')
                        with gr.Row():
                                cfg_interval_enable = gr.Checkbox(
                                value=False,
                                default=False,
                                label="Enable CFG Scheduler",
                                elem_id='cfg_interval_enable',
                                info="If Enabled and Schedule != Constant, applies CFG only within noise interval with the selected schedule type. SDXL recommend CFG=15; CFG interval (0.28, 5.42]"
                                )
                        with gr.Row():
                                cfg_schedule = gr.Dropdown(
                                        value='Constant',
                                        choices= SCHEDULES,
                                        label="CFG Schedule Type",
                                        elem_id='cfg_interval_schedule',
                                )
                                cfg_interval_low = gr.Slider(value = 0, minimum = 0, maximum = 100, step = 0.1, label="CFG Noise Interval Low", elem_id = 'cfg_interval_low', info="")
                                cfg_interval_high = gr.Slider(value = 100, minimum = 0, maximum = 100, step = 0.1, label="CFG Noise Interval High", elem_id = 'cfg_interval_high', info="")

                cfg_interval_enable.do_not_save_to_config = True
                cfg_schedule.do_not_save_to_config = True
                cfg_interval_low.do_not_save_to_config = True
                cfg_interval_high.do_not_save_to_config = True
                ep_cfg_enable.do_not_save_to_config = True
                self.infotext_fields = [
                        (cfg_interval_enable, lambda d: gr.Checkbox.update(value='CFG Interval Enable' in d)),
                        (cfg_schedule, 'CFG Interval Schedule'),
                        (cfg_interval_low, 'CFG Interval Low'),
                        (cfg_interval_high, 'CFG Interval High'),
                        (ep_cfg_enable, lambda d: gr.Checkbox.update(value='EP-CFG Enable' in d))
                ]
                self.paste_field_names = [
                        'cfg_interval_enable',
                        'cfg_interval_schedule',
                        'cfg_interval_low',
                        'cfg_interval_high',
                        'ep_cfg_interval_enable'
                ]
                return [cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, ep_cfg_enable]

        def process_batch(self, p: StableDiffusionProcessing, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, ep_cfg_enable, *args, **kwargs):
                # cleanup previous hooks always
                script_callbacks.remove_current_script_callbacks()
                self.remove_all_hooks()

                cfg_interval_enable = getattr(p, "cfg_interval_enable", cfg_interval_enable)
                ep_cfg_enable = getattr(p, "ep_cfg_enable", ep_cfg_enable)
                if cfg_interval_enable is False and ep_cfg_enable is False:
                        return
                cfg_schedule = getattr(p, "cfg_interval_schedule", cfg_schedule)
                cfg_interval_low = getattr(p, "cfg_interval_low", cfg_interval_low)
                cfg_interval_high = getattr(p, "cfg_interval_high", cfg_interval_high)
                if cfg_interval_enable:
                        p.extra_generation_params.update({
                                "CFG Interval Enable": cfg_interval_enable,
                                "CFG Interval Schedule": cfg_schedule,
                                "CFG Interval Low": cfg_interval_low,
                                "CFG Interval High": cfg_interval_high,
                        })
                if ep_cfg_enable:
                        p.extra_generation_params.update({
                                "EP-CFG Enable": ep_cfg_enable,
                        })
                self.create_hook(p,cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, ep_cfg_enable)

        def create_hook(self, p: StableDiffusionProcessing, cfg_interval_enable, cfg_schedule, cfg_interval_low, cfg_interval_high, ep_cfg_enable, *args, **kwargs):
                # Create a list of parameters for each concept
                cfgi_params = CFGSchedulerParams()

                # Add to p's incant_cfg_params
                if not hasattr(p, 'incant_cfg_params'):
                        logger.error("No incant_cfg_params found in p")
                p.incant_cfg_params['cfgi_params'] = cfgi_params

                cfgi_params.cfg_interval_enable = cfg_interval_enable
                cfgi_params.cfg_interval_schedule = cfg_schedule
                cfgi_params.max_sampling_step = p.steps
                cfgi_params.guidance_scale = p.cfg_scale
                cfgi_params.cfg_interval_scheduled_value = p.cfg_scale
                cfgi_params.ep_cfg_enable = ep_cfg_enable

                if cfgi_params.cfg_interval_enable:
                       # Refer to 3.1 Practice in the paper
                       # We want to round high and low noise levels to the nearest integer index
                       low_index = find_closest_index(cfg_interval_low, cfgi_params.max_sampling_step)
                       high_index = find_closest_index(cfg_interval_high, cfgi_params.max_sampling_step)
                       cfgi_params.cfg_interval_low = calculate_noise_level(low_index, cfgi_params.max_sampling_step)
                       cfgi_params.cfg_interval_high = calculate_noise_level(high_index, cfgi_params.max_sampling_step)
                       logger.debug(f"Step Aligned CFG Interval (low, high): ({low_index}, {high_index}), Step Aligned CFG Interval: ({round(cfgi_params.cfg_interval_low, 4)}, {round(cfgi_params.cfg_interval_high, 4)})")

                # Use lambda to call the callback function with the parameters to avoid global variables
                cfg_denoise_lambda = lambda callback_params: self.on_cfg_denoiser_callback(callback_params, cfgi_params)
                unhook_lambda = lambda _: self.unhook_callbacks(cfgi_params)

                script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
                script_callbacks.on_script_unloaded(unhook_lambda)

        def postprocess_batch(self, p, cfg_interval_enable, *args, **kwargs):
                script_callbacks.remove_current_script_callbacks()
                logger.debug('Removed script callbacks')
                active = getattr(p, "cfg_interval_enable", cfg_interval_enable)
                if active is False:
                        return

        def remove_all_hooks(self):
                return

        def unhook_callbacks(self, cfgi_params: CFGSchedulerParams):
                return

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, cfgi_params: CFGSchedulerParams):
                # always unhook
                self.unhook_callbacks(cfgi_params)

                cfgi_params.step = params.sampling_step
                # CFG Interval
                # TODO: set rho based on sdxl or sd1.5
                cfgi_params.current_noise_level = calculate_noise_level(
                        i = cfgi_params.step,
                        N = cfgi_params.max_sampling_step,
                )
                if cfgi_params.cfg_interval_enable:
                        if cfgi_params.cfg_interval_schedule != 'Constant':
                                # Calculate noise interval
                                start = cfgi_params.cfg_interval_low
                                end = cfgi_params.cfg_interval_high
                                begin_range = start if start <= end else end
                                end_range = end if start <= end else start
                                # Scheduled CFG Value
                                scheduled_cfg_scale = cfg_scheduler(cfgi_params.cfg_interval_schedule, cfgi_params.step, cfgi_params.max_sampling_step, cfgi_params.guidance_scale)
                                cfgi_params.cfg_interval_scheduled_value = scheduled_cfg_scale if begin_range <= cfgi_params.current_noise_level <= end_range else 1.0

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[CFG-SCHED] Enable CFG Scheduler", str, cfgs_apply_override('cfg_interval_enable', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[CFG-SCHED] CFG Noise Interval Low", float, cfgs_apply_field("cfg_interval_low")),
                        xyz_grid.AxisOption("[CFG-SCHED] CFG Noise Interval High", float, cfgs_apply_field("cfg_interval_high")),
                        xyz_grid.AxisOption("[CFG-SCHED] CFG Schedule Type", str, cfgs_apply_override('cfg_interval_schedule', boolean=False), choices=lambda: SCHEDULES),
                        xyz_grid.AxisOption("[CFG-SCHED] EP-CFG Enable", str, cfgs_apply_override('ep_cfg_enable', boolean=True), choices=xyz_grid.boolean_choice(reverse=True))
                }
                return extra_axis_options


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
def cfgs_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if 'cfg_interval_' in field and not hasattr(p, "cfg_interval_enable"):
            p.cfg_interval_enable = True
    return fun


def cfgs_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "cfg_interval_enable"):
                p.cfg_interval_enable = True
        setattr(p, field, x)
    return fun
