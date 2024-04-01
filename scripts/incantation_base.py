import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
from dataclasses import dataclass
from typing import Any

from modules import script_callbacks
from modules.processing import StableDiffusionProcessing
from scripts.ui_wrapper import UIWrapper
from scripts.incant import IncantExtensionScript
from scripts.t2i_zero import T2I0ExtensionScript
from scripts.pag import PAGExtensionScript

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))


"""

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""
class SubmoduleInfo:
        def __init__(self, module: UIWrapper, module_idx = 0, num_args = -1, arg_idx = -1):
                self.module: UIWrapper = module
                self.module_idx: int = num_args # the length of arg list
                self.num_args: int = num_args # the length of arg list
                self.arg_idx: int = arg_idx # where the list of args starts

submodules: list[SubmoduleInfo] = [
        SubmoduleInfo(module=PAGExtensionScript()),
        SubmoduleInfo(module=T2I0ExtensionScript()),
        SubmoduleInfo(module=IncantExtensionScript()),
]
                
class IncantBaseExtensionScript(scripts.Script):
        def __init__(self):
                pass

        # Extension title in menu UI
        def title(self):
                return "Incantations"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                # setup UI
                out = []
                with gr.Accordion('Incantations', open=False):
                        for idx, module_info in enumerate(submodules):
                                module_info.module_idx = idx
                                module = module_info.module
                                module_param_list = module.setup_ui(is_img2img)
                                module_info.num_args = len(module_param_list)
                                if module_info.num_args > 0:
                                        arg_idx = max(len(out), 0)
                                        module_info.arg_idx = arg_idx
                                        out.extend(module_param_list)
                # setup fields
                self.infotext_fields = []
                self.paste_field_names = []
                for module_info in submodules:
                        module = module_info.module
                        self.infotext_fields.extend(module.get_infotext_fields())
                        self.paste_field_names.extend(module.get_paste_field_names())
                return out
        
        def before_process(self, p: StableDiffusionProcessing, *args, **kwargs):
                for m in submodules:
                        m.module.before_process(p, *self.m_args(m, *args), **kwargs)

        def process(self, p: StableDiffusionProcessing, *args, **kwargs):
                for m in submodules:
                        m.module.process(p, *self.m_args(m, *args), **kwargs)

        def before_process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
                for m in submodules:
                        m.module.before_process_batch(p, *self.m_args(m, *args), **kwargs)
        
        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
                for m in submodules:
                        m.module.process_batch(p, *self.m_args(m, *args), **kwargs)

        def postprocess_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
                for m in submodules:
                        m.module.postprocess_batch(p, *self.m_args(m, *args), **kwargs)

        def unhook_callbacks(self):
                for m in submodules:
                        m.module.unhook_callbacks()
                script_callbacks.remove_current_script_callbacks()
        
        def m_args(self, module: SubmoduleInfo, *args):
                return args[module.arg_idx:module.arg_idx + module.num_args]


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def make_axis_options(extra_axis_options):
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
        current_opts = [x.label for x in xyz_grid.axis_options]
        # TODO: 
        for opt in extra_axis_options:
                if opt.label in current_opts:
                        return
        xyz_grid.axis_options.extend(extra_axis_options)


def callback_before_ui():
        try:
                for module_info in submodules:
                        module = module_info.module
                        extra_axis_options = module.get_xyz_axis_options()
                        make_axis_options(extra_axis_options)
        except:
                logger.exception("Incantation: Error while making axis options")

script_callbacks.on_before_ui(callback_before_ui)
