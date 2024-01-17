import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
from PIL import Image
import numpy as np
import re

from modules import script_callbacks, prompt_parser
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
from modules.prompt_parser import reconstruct_multicond_batch, stack_conds, reconstruct_cond_batch
from modules.processing import StableDiffusionProcessing, decode_latent_batch, txt2img_image_conditioning
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import pad_cond
from modules import shared, devices, errors
from modules.interrogate import InterrogateModels

import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""
!!!Currently non-functional!!!

An unofficial implementation of Seek for Incantations: Towards Accurate Text-to-Image Diffusion Synthesis
through Prompt Engineering for Automatic1111 WebUI

@misc{yu2024seek,
      title={Seek for Incantations: Towards Accurate Text-to-Image Diffusion Synthesis through Prompt Engineering}, 
      author={Chang Yu and Junran Peng and Xiangyu Zhu and Zhaoxiang Zhang and Qi Tian and Zhen Lei},
      year={2024},
      eprint={2401.06345},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""

class IncantStateParams:
        def __init__(self):
                self.coarse = 10
                self.fine = 30
                self.gamma = 0.25
                self.prompt = ''
                self.prompts = []
                self.prompt_tokens = []
                self.caption_coarse = []
                self.caption_fine = []
                self.img_coarse = []
                self.img_fine = []
                self.emb_img_coarse = []
                self.emb_img_fine = []
                self.emb_txt_coarse = []
                self.emb_txt_fine = []
                self.grad_txt = []
                self.grad_img = []
                self.matches_coarse = []
                self.matches_fine = []
                self.masked_prompt = []
                self.get_conds_with_caching = None
                self.steps = None
                self.iteration = None
                self.batch_size = 1
                self.p = None
                self.init_noise = None
                self.first_stage_cache = None
                self.second_stage = False
                self.denoiser = None
                self.job = None

                # is this loss
                self.loss = [] # grad_img * grad txt
                self.loss_qual = [] # quality guidance
                self.loss_sem = [] # semantic guidance
                self.loss_txt_txt = [] # txt - txt
                self.loss_txt_img = [] # txt - img
                self.loss_spar = [] # sparsity

                # hyperparameters
                self.qual_scale = 1.0
                self.sem_scale = 1.0
                self.txt_txt_scale = 1.0
                self.txt_img_scale = 1.0
                self.spar_scale = 1.0
                
class IncantExtensionScript(scripts.Script):
        def __init__(self):
                self.stage_1 = [[]]
                self.cached_c = [[None, None],[None, None]]

        # Extension title in menu UI
        def title(self):
                return "Incantations"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('Incantations', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='incant_active')
                        quality = gr.Checkbox(value=True, default=True, label="Append Prompt", elem_id='incant_quality')
                        with gr.Row():
                                coarse_step = gr.Slider(value = 10, minimum = 0, maximum = 100, step = 1, label="Coarse Step", elem_id = 'incant_coarse')
                                fine_step = gr.Slider(value = 30, minimum = 0, maximum = 100, step = 1, label="Fine Step", elem_id = 'incant_fine')
                                gamma = gr.Slider(value = 0.25, minimum = 0, maximum = 1.0, step = 0.01, label="Gamma", elem_id = 'incant_gamma')
                        with gr.Row():
                                qual_scale = gr.Slider(value = 0.0, minimum = 0, maximum = 100.0, step = 0.01, label="Quality Guidance Scale", elem_id = 'incant_qual_scale')
                                sem_scale = gr.Slider(value = 0.0, minimum = 0, maximum = 100.0, step = 0.01, label="Semantic Guidance Scale", elem_id = 'incant_sem_scale')
                                # txt_txt_scale = 1.0
                                # txt_img_scale = 1.0
                                # spar_scale = 1.0
                active.do_not_save_to_config = True
                quality.do_not_save_to_config = True
                coarse_step.do_not_save_to_config = True
                fine_step.do_not_save_to_config = True
                gamma.do_not_save_to_config = True
                qual_scale.do_not_save_to_config = True
                sem_scale.do_not_save_to_config = True
                # self.infotext_fields = [
                #         (active, lambda d: gr.Checkbox.update(value='INCANT Active' in d)),
                #         (coarse_step, 'INCANT Prompt'),
                #         (fine_step, 'INCANT Negative Prompt'),
                # ]
                # self.paste_field_names = [
                #         'incant_active',
                #         'incant_quality',
                #         'incant_coarse',
                #         'incant_fine',
                # ]
                return [active, quality, coarse_step, fine_step, gamma, qual_scale, sem_scale]

        def before_process(self, p: StableDiffusionProcessing, active, quality, coarse, fine, gamma, qual_scale, sem_scale, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                p.n_iter = p.n_iter * 2
        
        def process(self, p: StableDiffusionProcessing, active, quality, coarse, fine, gamma, qual_scale, sem_scale, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                quality = getattr(p, "incant_quality", quality)
                
                # modifying the all_prompts* may conflict with extensions that do so
                if p.iteration == 0:
                        param_list = [
                        # "prompts",
                        # "negative_prompts",
                        # "seeds",
                        # "subseeds",
                        "all_hr_negative_prompts",
                        "all_hr_prompts",
                        "all_negative_prompts",
                        "all_prompts",
                        "all_seeds",
                        "all_subseeds",
                        ]

                        for param_name in param_list:
                                run_fn_on_attr(p, param_name, duplicate_alternate_elements, p.batch_size)

                # assign
                        if quality:
                                for n in range(1, p.n_iter, 2):
                                        start_idx = n * p.batch_size
                                        end_idx = (n + 1) * p.batch_size
                                        p.all_prompts[start_idx:end_idx] = [prompt + ' BREAK <<REPLACEME>>' for prompt in p.all_prompts[start_idx:end_idx]]
                # elif p.iteration % 2 == 1:
                #         n = p.iteration
                #         start_idx = n * p.batch_size
                #         end_idx = (n + 1) * p.batch_size
                #         p.all_prompts[start_idx:end_idx] = [prompt.replace('<<REPLACEME>>', self.stage_1.masked_prompt) for prompt in p.all_prompts[start_idx:end_idx]]
                #         kwargs['prompts'] = [x.replace('<<REPLACEME>>', self.stage_1.masked_prompt) for x in kwargs['prompts']]
        

        # def before_process_batch(self, p: StableDiffusionProcessing, active, quality, coarse, fine, gamma, *args, **kwargs):
        #         active = getattr(p, "incant_active", active)
        #         if active is False:
        #                 return

                # TODO: Find more robust way to do this

                # Every even step will be when we use the previously calculated results
                # p.n_iter = p.n_iter * 2
                # print(f"n_iter: {p.n_iter}")

                # # Duplicate every element in each list if it exists
                # param_list = [
                #        "prompts",
                #        "negative_prompts",
                #        "seeds",
                #        "subseeds",
                #        "all_hr_negative_prompts",
                #        "all_hr_prompts",
                #        "all_negative_prompts",
                #        "all_prompts",
                #        "all_seeds",
                #        "all_subseeds",
                # ]
                # for param_name in param_list:
                #         run_fn_on_attr(p, param_name, duplicate_list)

        def before_process_batch(self, p: StableDiffusionProcessing, active, quality, coarse_step, fine_step, gamma, qual_scale, sem_scale, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                quality = getattr(p, "incant_quality", quality)
                coarse_step = getattr(p, "incant_coarse", coarse_step)
                fine_step = getattr(p, "incant_fine", fine_step)
                gamma = getattr(p, "incant_gamma", gamma)
                qual_scale = getattr(p, "incant_qual_scale", qual_scale)
                sem_scale = getattr(p, "incant_sem_scale", sem_scale)
                # if fine_step > p.steps:
                #         print(f"Fine step {fine_step} is greater than total steps {p.steps}, setting to {p.steps}")
                #         fine_step = p.steps

                interrogator = shared.interrogator
                interrogator.load()
                # Every even step will be when we use the previously calculated results
                # p.n_iter = p.n_iter * 2

                # modify prompts
                n = p.n_iter
                # print(f"n_iter: {p.n_iter}")

                # modify prompts such that every other prompt 
                # Duplicate every element in each list if it exists
                # if p.iteration == 0:
                #         param_list = [
                #         # "prompts",
                #         # "negative_prompts",
                #         # "seeds",
                #         # "subseeds",
                #         "all_hr_negative_prompts",
                #         "all_hr_prompts",
                #         "all_negative_prompts",
                #         "all_prompts",
                #         "all_seeds",
                #         "all_subseeds",
                #         ]

                #         for param_name in param_list:
                #                 run_fn_on_attr(p, param_name, duplicate_alternate_elements)

                # # assign
                #         for n in range(1, p.n_iter, 2):
                #                 start_idx = n * p.batch_size
                #                 end_idx = (n + 1) * p.batch_size
                #                 p.all_prompts[start_idx:end_idx] = [prompt + ' BREAK <<REPLACEME>>' for prompt in p.all_prompts[start_idx:end_idx]]
                if quality:
                        if p.iteration % 2 == 1:
                                n = p.iteration
                                start_idx = n * p.batch_size
                                end_idx = (n + 1) * p.batch_size
                                for idx in range(start_idx, end_idx):
                                        mask_idx = idx - start_idx
                                        p.all_prompts[idx] = p.all_prompts[idx].replace('<<REPLACEME>>', self.stage_1.masked_prompt[mask_idx])
                                        kwargs['prompts'][mask_idx] = kwargs['prompts'][mask_idx].replace('<<REPLACEME>>', self.stage_1.masked_prompt[mask_idx])

                # p.steps += fine_step
                # TODO: nicely put this into a dict
                p.extra_generation_params = {
                        "INCANT Active": active,
                        "INCANT Quality": quality,
                        "INCANT Coarse": coarse_step,
                        "INCANT Fine": fine_step,
                        "INCANT Gamma": gamma,
                        "INCANT Qual Scale": qual_scale,
                        "INCANT Sem Scale": sem_scale,
                }
                self.create_hook(p, active, quality, coarse_step, fine_step, gamma, qual_scale, sem_scale, *args, **kwargs)
        
        def process_batch(self, p: StableDiffusionProcessing, active, quality, coarse_step, fine_step, gamma, qual_scale, sem_scale, *args, **kwargs):

                batch_number = kwargs.get('batch_number', None)
                prompts = kwargs.get('prompts', None)
                seeds = kwargs.get('seeds', None)
                subseeds = kwargs.get('subseeds', None)
                gamma = getattr(p, 'incant_gamma', None)

                # if is second stage
                # if p.iteration % 2 == 1:
                #         if self.stage_1 is None:
                #                 print('\nerror: stage_1 is None')
                #         kwargs['prompts'] = [self.stage_1.masked_prompt for x in kwargs['prompts']]

        def parse_concept_prompt(self, prompt:str) -> list[str]:
                """
                Separate prompt by comma into a list of concepts
                TODO: parse prompt into a list of concepts using A1111 functions
                >>> g = lambda prompt: self.parse_concept_prompt(prompt)
                >>> g("")
                []
                >>> g("apples")
                ['apples']
                >>> g("apple, banana, carrot")
                ['apple', 'banana', 'carrot']
                """
                if len(prompt) == 0:
                        return []
                return [x.strip() for x in prompt.split(",")]

        def create_hook(self, p, active, quality, coarse, fine, gamma, qual_scale, sem_scale, *args, **kwargs):

                import clip
                # Create a list of parameters for each concept
                incant_params = IncantStateParams()
                incant_params.p = p
                incant_params.prompt = p.prompt
                incant_params.prompts = [pr for pr in p.prompts]
                #incant_params.prompt_tokens = clip.tokenize(list(p.prompt), truncate=True).to(devices.device_interrogate)
                incant_params.coarse = coarse
                incant_params.fine = fine 
                if fine > p.steps:
                        print(f"Fine step {fine} is greater than total steps {p.steps}, setting to {p.steps}")
                        fine = p.steps
                incant_params.gamma = gamma
                incant_params.qual_scale = qual_scale 
                incant_params.sem_scale = sem_scale 
                incant_params.iteration = p.iteration
                incant_params.get_conds_with_caching = p.get_conds_with_caching
                incant_params.steps = p.steps
                incant_params.batch_size = p.batch_size
                incant_params.job = shared.state.job
                #incant_params.first_stage_cache = self.stage_1[0]
                incant_params.second_stage = (p.iteration % 2) == 1
                tqdm = shared.total_tqdm

                if p.iteration % 2 == 0:
                        self.stage_1 = incant_params
                else:
                        # assign old cache to next iteration
                        incant_params.first_stage_cache = self.stage_1
                        # init interrogator
                        # self.interrogate_images(incant_params, p)
                        #try:
                        #        self.calc_quality_guidance(incant_params)
                        #except Exception as e:
                        #        print('\nexception when calculating quality guidance:\n')
                        #        print(e)


                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, incant_params)
                y2 = lambda params: self.cfg_after_cfg_callback(params, incant_params)
                #y3 = lambda params: self.cfg_after_cfg_callback(params, incant_params)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                #script_callbacks.on_cfg_denoised(y2)
                script_callbacks.on_cfg_after_cfg(y2)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)
        
        def calc_masked_prompt(self, incant_params: IncantStateParams, first_stage_cache):
                fs = first_stage_cache
                repl_word = '-'
                prompt = incant_params.p.prompt
                repl_threshold = incant_params.gamma * 100.0
                word_list = fs.matches_fine[0]
                masked_prompt = self.mask_prompt(repl_threshold, word_list, incant_params.p.prompt)
                return masked_prompt

        def calc_quality_guidance(self, incant_params: IncantStateParams):
                incant_params.loss_qual = []
                for i, (grad_img, grad_txt) in enumerate(zip(incant_params.grad_img, incant_params.grad_txt)):
                        incant_params.loss_qual.append(grad_img * grad_txt)

        def loss_sem(self, incant_params: IncantStateParams):
                p = incant_params.p
                caption = incant_params.prompt
                incant_params.loss_sem = []
                prompt_list = [caption] * p.batch_size
                prompts = prompt_parser.SdConditioning(prompt_list, width=p.width, height=p.height)
                c = incant_params.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, self.cached_c, p.extra_network_data)
                masked_prompts = incant_params.masked_prompt
                for i, (emb_fine, emb_coarse) in enumerate(zip(incant_params.emb_txt_fine, incant_params.emb_txt_coarse)):
                        incant_params.loss_sem.append(emb_fine - emb_coarse)
        
        def postprocess_batch(self, p, active, quality, coarse_step, fine_step, gamma, qual_scale, sem_scale, *args, **kwargs):

                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                interrogator = shared.interrogator
                interrogator.unload()
                # if self.stage_1 is not None:
                #         del self.stage_1
                #         self.stage_1 = None
                script_callbacks.remove_current_script_callbacks()

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, incant_params: IncantStateParams):
                second_stage = incant_params.second_stage
                if not second_stage:
                        return

                fs: IncantStateParams = incant_params.first_stage_cache

                # TODO: handle batches

                p = incant_params.p
                sampling_step = params.sampling_step
                text_cond = params.text_cond
                text_uncond = params.text_uncond
                gamma = incant_params.gamma * 100.0

                if incant_params.qual_scale != 0:
                        # quality guidance
                        grad_img_batch = []
                        grad_txt_batch = []
                        loss_qual_batch = []
                        for i in range(len(fs.emb_img_fine)):
                        #for i, (emb_fine, emb_coarse) in enumerate(zip(fs.emb_img_fine, fs.emb_img_coarse)):
                                img_fine = fs.emb_img_fine[i]
                                img_norm_fine = torch.norm(img_fine, dim=-1, keepdim=True)
                                img_coarse = fs.emb_img_coarse[i]
                                img_norm_coarse = torch.norm(img_coarse, dim=-1, keepdim=True)
                                grad_img = (img_fine / img_norm_fine**2) - (img_coarse / img_norm_coarse**2)
                                grad_img_batch.append(grad_img)

                        # compute the text guidance
                        #for i, (emb_fine, emb_coarse) in enumerate(zip(fs.emb_txt_fine, fs.emb_txt_coarse)):
                                #for b in range(incant_params.batch_size):

                                # not sure what to with batch, when is batch > 0? 
                                b = 0
                                txt_fine = fs.emb_txt_fine[i].batch[b][0].schedules[0].cond
                                txt_norm_fine = torch.norm(txt_fine, dim=-1, keepdim=True)
                                txt_coarse = fs.emb_txt_coarse[i].batch[b][0].schedules[0].cond
                                txt_norm_coarse = torch.norm(txt_coarse, dim=-1, keepdim=True)
                                grad_txt = (txt_fine / txt_norm_fine**2) - (txt_coarse / txt_norm_coarse**2)
                                grad_txt_batch.append(grad_txt)
                        
                        for i, (grad_img, grad_txt) in enumerate(zip(grad_img_batch, grad_txt_batch)):
                                loss_qual = grad_img * grad_txt
                                loss_qual_batch.append(loss_qual)
                                t = loss_qual * incant_params.qual_scale
                                print(f'\nloss_qual:{t.norm()}')
                                t = t.unsqueeze(0)

                                ## TODO: scale t by hyperparameter (and use correct formula)
                                if t.shape[1] != text_cond.shape[1]:
                                        empty = shared.sd_model.cond_stage_model_empty_prompt
                                        num_repeats = (t.shape[1] - text_cond.shape[1]) // empty.shape[1]

                                        if num_repeats < 0:
                                                t = pad_cond(t, -num_repeats, empty)
                                        elif num_repeats > 0:
                                                t = pad_cond(t, num_repeats, empty)
                                text_cond[i] += t.squeeze(0)



        def mask_prompt(self, gamma, word_list, prompt):
                regex = r"\b{0}\b"
                masked_prompt = prompt
                for word, pct in word_list: 
                        if pct < gamma:
                                repl_regex = regex.format(word)
                                        # replace word with -
                                masked_prompt = re.sub(repl_regex, "-", masked_prompt)
                return masked_prompt


                # if isinstance(text_cond, torch.Tensor) and isinstance(text_uncond, torch.Tensor):
                #         pass
                # else:
                #         raise NotImplementedError("Only SD1.5 are supported for now")

        def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams, incant_params: IncantStateParams):
                import clip
                p = incant_params.p
                coarse = incant_params.coarse
                fine = incant_params.fine
                second_stage = incant_params.second_stage
                x = params.x
                step = params.sampling_step
                max_step = params.total_sampling_steps

                # save the coarse images
                # this isn't quite the same thing
                # bc the paper says that the coarse image is the image where
                # the total steps is equal to coarse step
                if step == coarse and not second_stage:
                        print(f"\nCoarse step: {step}\n")
                        # decode the coarse latents
                        coarse_images = self.decode_images(x)
                        incant_params.img_coarse = coarse_images
                        devices.torch_gc()

                # FIXME: why is the max value of step 2 less than the total steps???
                elif step == fine - 2 and not second_stage:
                        print(f"\nFine step: {step}\n")

                        # decode fine images
                        fine_images = self.decode_images(x)
                        incant_params.img_fine = fine_images 

                        # compute embedding stuff
                        self.interrogate_images(incant_params, p)
                        devices.torch_gc()

                        # compute masked_prompts
                        for i, matches in enumerate(incant_params.matches_fine):
                                incant_params.masked_prompt.append(self.mask_prompt(incant_params.gamma*100.0, matches, p.prompt))

                        # calculate gradients
                        # self.calculate_embedding_gradients(incant_params, p, step)

                        # calculate quality guidance
                        # try:
                        #         self.calc_quality_guidance(incant_params)
                        # except Exception as e:
                        #         print('\nexception when calculating quality guidance:\n')
                        #         print(e)
                        
                else:
                        pass

        def compute_gradients(self, emb_fine, emb_coarse):
                out_gradients = []
                # zip together list and iterate
                for i, (fine, coarse) in enumerate(zip(emb_fine, emb_coarse)):
                        # calculate norm of fine and coarse embeddings
                        norm_fine = torch.norm(fine, dim=-1, keepdim=True)
                        norm_fine **= 2
                        norm_coarse = torch.norm(coarse, dim=-1, keepdim=True)
                        norm_coarse **= 2
                        grad = (fine/norm_fine) - (coarse/norm_coarse)
                        out_gradients.append(grad)
                return out_gradients

        def calculate_embedding_gradients(self, incant_params, p, current_step):
                # text embeddings
                captions_coarse = incant_params.caption_coarse
                captions_fine = incant_params.caption_fine
                # txt_emb_coarse = incant_params.emb_txt_coarse
                # txt_emb_fine = incant_params.emb_txt_fine
                # txt_emb_fine = []
                # txt_emb_coarse = []
                for i in range(len(captions_coarse)):
                        out = []
                        incant_params.grad_txt.append(out)
                # image embeddings
                img_emb_coarse = incant_params.emb_img_coarse
                img_emb_fine = incant_params.emb_img_fine
                # incant_params.grad_img = self.compute_gradients(img_emb_fine, img_emb_coarse)

        def interrogate_images(self, incant_params, p):
                interrogator = shared.interrogator
                interrogator.load()

                # calculate text/image embeddings
                text_array = incant_params.prompt.split()

                #shared.state.begin(job="interrogate")
                # coarse features
                # for refactoring later
                img_list = incant_params.img_coarse
                caption_list = incant_params.caption_coarse
                clip_img_embed_list = incant_params.emb_img_coarse
                cond_list = incant_params.emb_txt_coarse
                matches_list = incant_params.matches_coarse

                for i, pil_image in enumerate(img_list):
                        caption = interrogator.generate_caption(pil_image)
                        caption_list.append(caption)

                        devices.torch_gc()
                        res = caption
                        clip_image = interrogator.clip_preprocess(pil_image).unsqueeze(0).type(interrogator.dtype).to(devices.device_interrogate)
                        with torch.no_grad(), devices.autocast():
                                # calculate image embeddings
                                image_features = interrogator.clip_model.encode_image(clip_image).type(interrogator.dtype)
                                # image_features /= image_features.norm(dim=-1, keepdim=True)
                                clip_img_embed_list.append(image_features)

                                # calculate text embeddings
                                prompt_list = [caption]
                                #prompt_list = [caption] * p.batch_size
                                prompts = prompt_parser.SdConditioning(prompt_list, width=p.width, height=p.height)
                                c = incant_params.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, self.cached_c, p.extra_network_data)
                                cond_list.append(c)

                                # calculate image similarity
                                matches = interrogator.rank(image_features, text_array, top_count=len(text_array))
                                print(f"{i}-caption:{caption}\n{i}-coarse: {matches}")
                                matches_list.append(matches)

                # fine features
                for i, pil_image in enumerate(incant_params.img_fine):
                        caption = interrogator.generate_caption(pil_image)
                        incant_params.caption_fine.append(caption)

                        devices.torch_gc()
                        res = caption
                        clip_image = interrogator.clip_preprocess(pil_image).unsqueeze(0).type(interrogator.dtype).to(devices.device_interrogate)

                        with torch.no_grad(), devices.autocast():
                                # calculate image embeddings
                                image_features = interrogator.clip_model.encode_image(clip_image).type(interrogator.dtype)
                                image_features /= image_features.norm(dim=-1, keepdim=True)
                                incant_params.emb_img_fine.append(image_features)

                                # calculate text embeddings
                                prompt_list = [caption]
                                #prompt_list = [caption] * p.batch_size
                                prompts = prompt_parser.SdConditioning(prompt_list, width=p.width, height=p.height)
                                c = incant_params.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, self.cached_c, p.extra_network_data)
                                incant_params.emb_txt_fine.append(c)

                                # calculate image similarity
                                matches = interrogator.rank(image_features, text_array, top_count=len(text_array))
                                incant_params.matches_fine.append(matches)
                                print(f"{i}-caption:{caption}\n{i}-fine:{matches}")
                devices.torch_gc()

        def decode_images(self, x):
                batch_images = []
                already_decoded = False
                x_samples_ddim = decode_latent_batch(shared.sd_model, x, target_device=devices.cpu, check_for_nans=True)
                for i, x_sample in enumerate(x_samples_ddim):
                        x_sample = x_sample.to(torch.float32)
                        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                        x_sample = x_sample.astype(np.uint8)
                        image = Image.fromarray(x_sample)
                        batch_images.append(image)
                return batch_images


def run_fn_on_attr(p, attr_name, fn, *args):
        """ Run a function on an attribute of a class if it exists """
        try:
                attr = getattr(p, attr_name)
                setattr(p, attr_name, fn(attr, *args))
        except AttributeError:
                # No attribute exists
                return
        except TypeError:
                # If the attribute is not iterable, return
                return

def duplicate_alternate_elements(input_list: list, batch_size = 1) -> list:
        """ Duplicate each element in a list and return a new list
        >>> duplicate_list([1, 2, 3, 4], 1)
        [1, 1, 3, 3]
        >>> duplicate_list([1, 2, 3, 4], 2)
        [1, 2, 1, 2]
        >>> duplicate_list([1, 2, 3, 4, 5, 6, 7, 8], 4)
        [1, 2, 3, 4, 1, 2, 3, 4]
        """
        result = []
        for i in range(0, len(input_list), batch_size*2):
                batch = input_list[i:i + batch_size]
                result.extend(batch)
                result.extend(batch)
        return result

def duplicate_list(input_list: list) -> list:
        """ Duplicate each element in a list and return a new list
        >>> duplicate_list([1,2,3])
        [1, 1, 2, 2, 3, 3]
        """
        return [element for item in input_list for element in (item, item)]


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def incant_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def incant_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "incant_active"):
                setattr(p, "incant_active", True)
        setattr(p, field, x)

    return fun

def make_axis_options():
        xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
        extra_axis_options = {
                xyz_grid.AxisOption("[Incant] Active", str, incant_apply_override('incant_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[Incant] Quality", str, incant_apply_override('incant_quality', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                xyz_grid.AxisOption("[Incant] Coarse Step", int, incant_apply_field("incant_coarse")),
                xyz_grid.AxisOption("[Incant] Fine Step", int, incant_apply_field("incant_fine")),
                #xyz_grid.AxisOption("[Incant] Warmup Steps", int, incant_apply_field("incant_warmup")),
                #xyz_grid.AxisOption("[Incant] Guidance Scale", float, incant_apply_field("incant_edit_guidance_scale")),
                #xyz_grid.AxisOption("[Incant] Tail Percentage Threshold", float, incant_apply_field("incant_tail_percentage_threshold")),
                #xyz_grid.AxisOption("[Incant] Momentum Scale", float, incant_apply_field("incant_momentum_scale")),
                #xyz_grid.AxisOption("[Incant] Momentum Beta", float, incant_apply_field("incant_momentum_beta")),
        }
        if not any("[Incant]" in x.label for x in xyz_grid.axis_options):
                xyz_grid.axis_options.extend(extra_axis_options)

def callback_before_ui():
        try:
                make_axis_options()
        except:
                logger.exception("Incantation: Error while making axis options")

script_callbacks.on_before_ui(callback_before_ui)
