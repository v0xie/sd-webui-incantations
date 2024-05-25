import gradio as gr
from scripts.ui_wrapper import UIWrapper
from scripts.incant_utils import module_hooks

"""
WIP Implementation of https://arxiv.org/abs/2404.11824
Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""
class TCGExtensionScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []

    def title(self) -> str:
        raise 'TCG [arXiv:2404.11824]'
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion('TCG', open=True):
            active = gr.Checkbox(label="Active", value=True)
            opts = [active]
            for opt in opts:
                opt.do_not_save_to_config = True
            return opts

    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def before_process_batch(self, p, *args, **kwargs):
        pass

    def process_batch(self, p, *args, **kwargs):
        pass

    def postprocess_batch(self, p, *args, **kwargs):
        pass
    
    def unhook_callbacks(self) -> None:
        pass

    def get_xyz_axis_options(self) -> dict:
        return {}

def arg(p, field_name: str, variable_name:str, default=None, **kwargs):
        """ Get argument from field_name or variable_name, or default if not found """
        return getattr(p, field_name, kwargs.get(variable_name, None))
