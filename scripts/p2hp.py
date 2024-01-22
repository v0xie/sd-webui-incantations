from modules import shared
import gradio as gr
from scripts.ui_wrapper import UIWrapper

class P2HP(UIWrapper):
    def __init__(self, config):
        self.infotext_fields: list = []
        self.paste_field_names: list = []
        self.config = config
        self.lr = 1e-02
    
    def setup(self):
        model = shared.sd_model
        conditioner = model.get_learned_conditioning

    def title(self) -> str:
        return "P2HP"
    
    def setup_ui(self, is_img2img) -> list:
        return self.ui(is_img2img)
    
    def ui(self, is_img2img) -> list:
        with gr.Accordion('P2HP', open=False):
            lr = gr.Slider(value=1e-02, default=1e-02, label='p2hp_lr', default='1e-02')
            out = [lr] 
            for p in out:
                p.do_not_save_to_config = True
            return out


    def get_infotext_fields(self) -> list:
        return self.infotext_fields

    def get_paste_field_names(self) -> list:
        return self.paste_field_names
    
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

