from modules import shared, interrogate, devices
from modules.shared import opts
import k_diffusion
import tqdm 
import gradio as gr
import torch
from scripts.ui_wrapper import UIWrapper
from scripts.prompt_optim_utils import optimize_prompt, optimize_prompt_s4a
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
        with gr.Accordion('Prompt Optimization', open=False):
            #img_path = "F:\\temp\\meme.png"
            with gr.Row():
                img = gr.Image(value=None, label='p2hp_img', sources=['upload','clipboard'], type='pil')
                img_coarse = gr.Image(value=None, label='p2hp_img_coarse', sources=['upload','clipboard'], type='pil')
            with gr.Row():
                input_prompt = gr.Textbox(value="", label='p2hp_input')
                output = gr.Textbox(value="", label='p2hp_output')
            prompt_len = gr.Slider(value=16, default=16, maximum=77, minimum=5, step = 1, label='p2hp_prompt_len')
            lr = gr.Slider(value=1e-02, default=1e-02, maximum=0.2, minimum=1e-03, step = 0.001, label='p2hp_lr')
            iterations = gr.Slider(value=2000, default=2000, maximum=10000, minimum=100, step = 100, label='p2hp_iter')
            steps = gr.Slider(value=100, default=100, step=1, label='p2hp_steps')
            batch_size = gr.Slider(value=1, default=1, maximum=16, minimum=1, step=1, label='p2hp_bs')
            loss_weight = gr.Slider(value=1.0, default=1.0, step=0.01, minimum=0, maximum=2, label='p2hp_loss_weight')
            loss_qual = gr.Slider(value=0.5, default=0.5, step=0.01, minimum=-2, maximum=2, label='p2hp_loss_qual')
            loss_sem = gr.Slider(value=1.0, default=1.0, step=0.01, minimum=-2, maximum=2, label='p2hp_loss_sem')
            loss_tt = gr.Slider(value=0.6, default=0.6, step=0.01, minimum=-2, maximum=2, label='p2hp_loss_tt')
            loss_spar = gr.Slider(value=0.5, default=0.5, step=0.01, minimum=-2, maximum=2, label='p2hp_loss_sparsity')
            loss_ti = gr.Slider(value=0.8, default=0.8, step=0.01, minimum=-2, maximum=2, label='p2hp_loss_ti')
            btn = gr.Button(value='Pez', type='button')
            btn.click(self.call_optimize_prompt, inputs = [img, img_coarse, prompt_len, lr, iterations, steps, batch_size, input_prompt, loss_weight, loss_qual, loss_sem, loss_tt, loss_spar, loss_ti], outputs = [output])

            out = [img, img_coarse, prompt_len, lr, iterations, steps, batch_size, input_prompt, loss_weight, loss_qual, loss_sem, loss_tt, loss_spar, loss_ti, btn] 
            for p in out:
                p.do_not_save_to_config = True

            return out
        
    def call_optimize_prompt(self, img, img_coarse, prompt_len, lr, iter, steps, batch_size, input_prompt, loss_weight, loss_qual, loss_sem, loss_tt, loss_spar, loss_ti):
        print('Calling p2hp')

        if img is None:
            return

        run_args = {
            'prompt_len': prompt_len,
            'lr': lr,
            'iter': iter,
            'print_step': steps,
            'batch_size': batch_size,
            'prompt_bs': batch_size,
            'weight_decay': 0.01, #TODO: make this a parameter
            'print_new_best': False, 
            'print_new_best_sum': True, 
            'loss_weight': loss_weight,
            'loss_qual': loss_qual,
            'loss_sem': loss_sem,
            'loss_tt': loss_tt,
            'loss_ti': loss_ti,
            'loss_spar': loss_spar,
        }
        input_prompt = input_prompt.strip()
        target_prompts = None if len(input_prompt) == 0 else [input_prompt]

        if img_coarse is not None:
            print('Optimizing prompt with coarse image')
            learned_prompt = optimize_prompt_s4a(input_prompt, [img_coarse], [img], run_args = run_args)
        else:
            learned_prompt = optimize_prompt(device=shared.device, args=run_args, target_images=[img], target_prompts=target_prompts)
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
