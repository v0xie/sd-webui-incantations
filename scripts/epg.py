import logging
from os import environ
import torch
import gradio as gr

from modules import script_callbacks, shared, patches
from modules import processing
from scripts.ui_wrapper import UIWrapper
from scripts.incant_utils import module_hooks

"""

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
        active = getattr(p, "epg_active", active)
        if not active:
            return
        tau = getattr(p, "epg_tau", temperature)
        start_idx = getattr(p, "epg_start_idx", start_idx)
        end_idx = getattr(p, "epg_end_idx", end_idx)
        start_step = getattr(p, "epg_start_step", start_step)

        # Hooks
        def pl_forward_hook(module, input, kwargs, output):
            output[0][0] *= tau
            return output

        def epg_get_conds_with_caching_wrapper(*args, **kwargs):
            # run on negative only
            prompts = args[1]
            if not prompts.is_negative_prompt:
                return self.og_func(*args, **kwargs)
            # patch
            handles = []
            crossattn_modules = self.get_crossattn_modules()
            for i, module in enumerate(crossattn_modules):
                if 5 < i < 10: 
                    module_hooks.module_add_forward_hook(module, pl_forward_hook, hook_type='forward', with_kwargs=True)
                    handles.add(module)
                    logger.debug(f"EPG: Added forward hook to {i}: {module}")
            if not crossattn_modules:
                logger.error("No self attention modules found, cannot run")
            if not self.og_func:
                logger.error("EPG: get_conds_with_caching_wrapper called without original function")
            # call the original function
            output = self.og_func(*args, **kwargs)
            # unpatch
            for handle in handles:
                module_hooks.remove_module_forward_hook(handle, 'pl_forward_hook')
            return output

        # patch the original function
        if not self.og_func:
            self.og_func = p.get_conds_with_caching
        p.get_conds_with_caching = epg_get_conds_with_caching_wrapper

    def get_crossattn_modules(self):
        crossattn_modules = module_hooks.get_modules(
             network_layer_name_filter='transformer_text_model_encoder_layers',
             module_name_filter='CLIPAttention'
        )
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
        return {}

