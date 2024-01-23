from modules import shared, interrogate
import gradio as gr
import torch
from scripts.ui_wrapper import UIWrapper
from PIL import Image
from modules.sd_samplers_common import images_tensor_to_samples, decode_first_stage, approximation_indexes

class P2HP(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []
        self.lr = 1e-02
        self.num_steps = 50
    
    def title(self) -> str:
        return "P2HP"
    
    def setup_ui(self, is_img2img) -> list:
        return self.ui(is_img2img)
    
    def ui(self, is_img2img) -> list:
        with gr.Accordion('P2HP', open=False):
            img_path = "F:\\temp\\meme.png"
            with gr.Row():
                img = gr.Image(value=img_path, label='p2hp_img', sources=['upload','clipboard'], type='pil', default=img_path)
            with gr.Row():
                output = gr.Textbox(value="", label='p2hp_output')
            lr = gr.Slider(value=1e-02, default=1e-02, step = 0.01, label='p2hp_lr')
            steps = gr.Slider(value=100, default=100, step=1, label='p2hp_steps')
            btn = gr.Button(label='P2HP', type='button')
            btn.click(self.p2hp, inputs = [img, lr], outputs = [output])
            out = [img, lr, btn] 
            for p in out:
                p.do_not_save_to_config = True
            return out
        
    def p2hp(self, img, lr):
        print('Calling p2hp')
        interrogator = shared.interrogator
        interrogator.load()

        prompt = "empty prompt"
        model = shared.sd_model
        conditioner = model.get_learned_conditioning(prompt)

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

