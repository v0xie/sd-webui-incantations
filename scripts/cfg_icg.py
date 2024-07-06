import os
import gradio as gr
from scripts.ui_wrapper import UIWrapper
from modules import processing, script_callbacks, scripts, rng
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import on_cfg_denoiser, CFGDenoiserParams
from modules.rng import randn_local

import torch


"""
An unofficial implementation of "No Training, No Problem: Rethinking Classifier-Free
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

class ICGStateParams:
    def __init__(self):
        self.uncond = None


class ICGExtensionScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []

    def title(self) -> str:
        return "Independent Condition Guidance"
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion("Independent Condition Guidance", open=False):
            active = gr.Checkbox(label="Active", default=True, elem_id="icg_active")
        active.do_not_save_to_config = True
        
        self.infotext_fields = [
                (active, lambda d: gr.Checkbox.update(value='ICG Active' in d)),
        ]
        self.paste_field_names = [
                'icg_active',
        ]
        return [active]

    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def before_process_batch(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        self.unhook_callbacks()
        active = getattr(p, "icg_active", active)

        if not active:
            return

        p.extra_generation_params.update({
            "ICG Active": active,
        })

        icg_params = ICGStateParams()

        def cfg_denoiser_callback(params: CFGDenoiserParams, icg_params: ICGStateParams):
            # replace uncond with random conditioning vector
            #if icg_params.uncond is None:
            #    uncond = rng.randn_local(p.seed, params.text_uncond.shape)
            #    icg_params.uncond = uncond
            new_uncond = rng.randn_like(params.text_uncond) * params.text_uncond.std()
            params.text_uncond = new_uncond
            #params.text_uncond = icg_params.uncond

        on_cfg_denoiser(lambda params: cfg_denoiser_callback(params, icg_params))

    def process_batch(self, p, *args, **kwargs):
        pass

    def postprocess_batch(self, p, *args, **kwargs):
        pass
    
    def unhook_callbacks(self) -> None:
        script_callbacks.remove_current_script_callbacks()

    def get_xyz_axis_options(self) -> dict:
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
        extra_axis_options = {
                xyz_grid.AxisOption("[ICG] Active", str, icg_apply_override('icg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
        }
        return extra_axis_options

def arg(p, field_name: str, variable_name:str, default=None, **kwargs):
        """ Get argument from field_name or variable_name, or default if not found """
        return getattr(p, field_name, kwargs.get(variable_name, None))

# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def icg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
        if not hasattr(p, "icg_active"):
                setattr(p, "icg_active", True)
    return fun

def icg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "icg_active"):
                setattr(p, "icg_active", True)
        setattr(p, field, x)
    return fun
