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
        return "P2HP"
    
    def setup_ui(self, is_img2img) -> list:
        return self.ui(is_img2img)
    
    def ui(self, is_img2img) -> list:
        with gr.Accordion('P2HP', open=True):
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
        learned_prompt = optimize_prompt(device=shared.device, target_images=[img])
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
    
# modified from modules/sd_samples_timesteps_impl.py
@torch.no_grad()
def ddim(model, x, timesteps, extra_args=None, callback=None, disable=None, eta=0.0):
    alphas_cumprod = model.inner_model.inner_model.alphas_cumprod
    alphas = alphas_cumprod[timesteps]
    alphas_prev = alphas_cumprod[torch.nn.functional.pad(timesteps[:-1], pad=(1, 0))].to(torch.float64 if x.device.type != 'mps' and x.device.type != 'xpu' else torch.float32)
    sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
    sigmas = eta * np.sqrt((1 - alphas_prev.cpu().numpy()) / (1 - alphas.cpu()) * (1 - alphas.cpu() / alphas_prev.cpu().numpy()))

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones((x.shape[0]))
    s_x = x.new_ones((x.shape[0], 1, 1, 1))
    for i in tqdm.trange(len(timesteps) - 1, disable=disable):
        index = len(timesteps) - 1 - i

        e_t = model(x, timesteps[index].item() * s_in, **extra_args)

        a_t = alphas[index].item() * s_x
        a_prev = alphas_prev[index].item() * s_x
        sigma_t = sigmas[index].item() * s_x
        sqrt_one_minus_at = sqrt_one_minus_alphas[index].item() * s_x

        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t
        noise = sigma_t * k_diffusion.sampling.torch.randn_like(x)
        x = a_prev.sqrt() * pred_x0 + dir_xt + noise

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': 0, 'sigma_hat': 0, 'denoised': pred_x0})

    return x

