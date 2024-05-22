import gradio as gr
from scripts.ui_wrapper import UIWrapper
from modules.processing import StableDiffusionProcessing
from modules import script_callbacks
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
import torch

class InitnoScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []
        self.cached_c = [None, None]

    def title(self) -> str:
        return "Embeds"
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion(label="Embeds", open=False):
            active = gr.Checkbox(label="Active", default=True, elem_id='embeds_active')
        return [active]

    def get_infotext_fields(self) -> list:
        return self.infotext_fields

    def get_paste_field_names(self) -> list:
        return self.paste_field_names
    
    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def before_process_batch(self, p, *args, **kwargs):
        self.unhook_callbacks()

    def process_batch(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        active = getattr(p, 'embeds_active', active)
        if not active:
            return

        def on_cfg_denoiser(params: CFGDenoiserParams):
            pass
            
        script_callbacks.on_cfg_denoiser(lambda x: on_cfg_denoiser(x))

    def optimize_noise(self, x):
        """ Optimize noise for a given image.
        Arguments:
            x: input initial latent
        """
        max_step = 50
        max_round = 5
        for step in range(max_step):
            for round in range(max_round):
                pass

    def postprocess_batch(self, p, *args, **kwargs):
        pass
    
    def unhook_callbacks(self) -> None:
        script_callbacks.remove_current_script_callbacks()

    def get_xyz_axis_options(self) -> dict:
        return {}
