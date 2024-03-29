from modules import shared, interrogate, devices
from modules.shared import opts
import k_diffusion
import tqdm 
import gradio as gr
import torch
from scripts.ui_wrapper import UIWrapper
from scripts.prompt_optim_utils import optimize_prompt
import numpy as np
from PIL import Image
from modules.sd_samplers_common import images_tensor_to_samples, decode_first_stage, approximation_indexes

class P2HP(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []
        self.lr = 1e-02
        self.num_steps = 50
        interrogator = shared.interrogator
        interrogator.load()
    
    def title(self) -> str:
        return "Prompt Optimization"
    
    def setup_ui(self, is_img2img) -> list:
        return self.ui(is_img2img)
    
    def ui(self, is_img2img) -> list:
        with gr.Accordion('Prompt Optimization', open=True):
            #img_path = "F:\\temp\\meme.png"
            with gr.Row():
                img = gr.Image(value=None, label='p2hp_img', sources=['upload','clipboard'], type='pil')
            with gr.Row():
                output = gr.Textbox(value="", label='p2hp_output')
            lr = gr.Slider(value=1e-02, default=1e-02, maximum=0.2, minimum=1e-03, step = 0.001, label='p2hp_lr')
            iterations = gr.Slider(value=3000, default=3000, maximum=10000, minimum=100, step = 100, label='p2hp_iter')
            steps = gr.Slider(value=100, default=100, step=1, label='p2hp_steps')
            batch_size = gr.Slider(value=1, default=1, maximum=16, minimum=1, step=1, label='p2hp_bs')
            btn = gr.Button(value='Pez', type='button')
            btn.click(self.call_optimize_prompt, inputs = [img, lr, iterations, steps, batch_size], outputs = [output])

            out = [img, lr, iterations, steps, batch_size, btn] 
            for p in out:
                p.do_not_save_to_config = True

            return out
        
    def call_optimize_prompt(self, img, lr, iter, steps, batch_size):
        print('Calling p2hp')

        if img is None:
            return

        run_args = {
            'lr': lr,
            'iter': iter,
            'steps': steps,
            'batch_size': batch_size
        }


        learned_prompt = optimize_prompt(device=shared.device, args=run_args, target_images=[img])
        return learned_prompt

        # interrogator = shared.interrogator
        # interrogator.load()

        # prompt = "empty prompt"
        # model = shared.sd_model
        # #conditioner = model.get_learned_conditioning(prompt)
        # image = np.asarray(img)
        # image = torch.from_numpy(image)
        # image = image.to(shared.device, dtype=devices.dtype_vae)

        # self.init_latent = images_tensor_to_samples(image, None, model)

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
