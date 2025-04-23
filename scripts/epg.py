import logging
from os import environ
import torch
import gradio as gr

from modules import scripts, script_callbacks, shared, patches
from modules import processing
from modules.processing import StableDiffusionProcessing
from scripts.ui_wrapper import UIWrapper
from scripts.incant_utils import module_hooks

"""
An unofficial implementation of ERG (Entropy Rectifying Guidance) for stable-diffusion-webui.

@misc{ifriqi2025entropyrectifyingguidancediffusion,
      title={Entropy Rectifying Guidance for Diffusion and Flow Models}, 
      author={Tariq Berrada Ifriqi and Adriana Romero-Soriano and Michal Drozdzal and Jakob Verbeek and Karteek Alahari},
      year={2025},
      eprint={2504.13987},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.13987}, 
}

"""

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))
incantations_debug = environ.get("INCANTAIONS_DEBUG", False)


class EPGExtensionScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []
        self.og_func = None

    def title(self) -> str:
        return "EPG"
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                active = gr.Checkbox(label="Active", value=True, elem_id="epg_active")
                temperature = gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.01, step=0.01, elem_id="epg_tau")
            with gr.Row():
                start_idx = gr.Slider(label="Start Index", minimum=0, maximum=300, value=5, step=1, elem_id="epg_start_idx")
                end_idx = gr.Slider(label="End Index", minimum=0, maximum=300, value=10, step=1, elem_id="epg_end_idx")
            with gr.Row():
                start_step = gr.Slider(label="Start Step", minimum=0, maximum=300, value=5, step=1, elem_id="epg_start_step")

        params = [active, temperature, start_idx, end_idx, start_step]
        for p in params:
            p.do_not_save_to_config = True
        return params

    def get_infotext_fields(self) -> list:
        return self.infotext_fields

    def get_paste_field_names(self) -> list:
        return self.paste_field_names
    
    def before_process(self, p, active, *args, **kwargs):
        pass

    def process(self, p, active, *args, **kwargs):
        pass

    def before_process_batch(self, p, active, *args, **kwargs):
        pass

    def process_before_every_sampling(self, p, active, *args, **kwargs):
        pass

    def process_batch(self, p, active,temperature, start_idx, end_idx, start_step, *args, **kwargs):
        # hook before setup_conds in modules.processing
        script_callbacks.remove_current_script_callbacks()
        self.unhook_callbacks()

        #p.cached_uc = [None, None]  # reset cached uc
        active = getattr(p, "epg_active", active)
        if not active:
            if self.og_func:
                p.get_conds_with_caching = self.og_func
                StableDiffusionProcessing.cached_uc = [None, None]
            self.og_func = None
            return
        tau = getattr(p, "epg_tau", temperature)
        start_idx = getattr(p, "epg_start_idx", start_idx)
        end_idx = getattr(p, "epg_end_idx", end_idx)
        start_step = getattr(p, "epg_start_step", start_step)

        # Hooks
        def pl_forward_hook(module, input, kwargs, output):
            output[:] *= tau
            return output

        def epg_get_conds_with_caching_wrapper(*args, **kwargs):
            # run on negative only
            prompts = args[1]
            if not prompts.is_negative_prompt:
                return self.og_func(*args, **kwargs)
            # jank af fix for caching
            #if not hasattr(prompts, "epg"):
            #    #p.cached_c = [None, None] # epg doesn't run on positive
            #    p.cached_uc = [None, None]
            prompts.epg = True
            # patch
            handles = []
            crossattn_modules = self.get_crossattn_modules()
            for i, module in enumerate(crossattn_modules):
                #if 5 < i < 10: 
                    module_hooks.module_add_forward_hook(module, pl_forward_hook, hook_type='forward', with_kwargs=True)
                    handles.append(module)
                    logger.debug(f"EPG: Added forward hook to {i}: {module.network_layer_name}")
            if not crossattn_modules:
                logger.error("No self attention modules found, cannot run")
            if not self.og_func:
                logger.error("EPG: get_conds_with_caching_wrapper called without original function")
            # call the original function
            output = self.og_func(args[0], args[1], args[2], [[None, None]], args[4])
            #output = self.og_func(*args, **kwargs)
            # unpatch
            for handle in handles:
                module_hooks.remove_module_forward_hook(handle, 'pl_forward_hook')
                logger.debug(f"EPG: Removed forward hook from {handle.network_layer_name}")
            return output

        # patch the original function
        if not self.og_func:
            self.og_func = p.get_conds_with_caching
        p.get_conds_with_caching = epg_get_conds_with_caching_wrapper

    def get_crossattn_modules(self):
        crossattn_modules = module_hooks.get_modules( network_layer_name_filter='transformer_text_model_encoder_layers', module_name_filter='Linear')
        crossattn_modules = [x for x in crossattn_modules if x.network_layer_name.endswith('self_attn_q_proj')]
        return crossattn_modules

    def postprocess_batch(self, p, active, *args, **kwargs):
        pass
    
    def unhook_callbacks(self) -> None:
        if self.og_func:
            pass
        crossattn_modules = self.get_crossattn_modules()
        for module in crossattn_modules:
            module_hooks.remove_module_forward_hook(module, 'pl_forward_hook')
        pass

    def get_xyz_axis_options(self) -> dict:
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
        extra_axis_options = {
                xyz_grid.AxisOption("[EPG] Enable EPG", str, epg_apply_override('epg_enable', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                # xyz_grid.AxisOption("[CFG-SCHED] CFG Noise Interval Low", float, cfgs_apply_field("cfg_interval_low")),
                # xyz_grid.AxisOption("[CFG-SCHED] CFG Noise Interval High", float, cfgs_apply_field("cfg_interval_high")),
                # xyz_grid.AxisOption("[CFG-SCHED] CFG Schedule Type", str, cfgs_apply_override('cfg_interval_schedule', boolean=False), choices=lambda: SCHEDULES),
                # xyz_grid.AxisOption("[CFG-SCHED] EP-CFG Enable", str, cfgs_apply_override('ep_cfg_enable', boolean=True), choices=xyz_grid.boolean_choice(reverse=True))
        }
        return extra_axis_options

# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def epg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if 'epg_' in field and not hasattr(p, "epg_enable"):
            p.epg_enable = True
    return fun


def epg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "epg_enable"):
                p.epg_enable = True
        setattr(p, field, x)
    return fun
