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
from modules import shared, devices, errors, deepbooru
from modules.interrogate import InterrogateModels
from scripts.ui_wrapper import UIWrapper, arg
# from scripts.t2i_zero import SegaExtensionScript

import torch
from torchvision.transforms import ToPILImage

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))


"""
!!!
!!! Only semi-functional !!!
!!!

!!! Might conflict with other extensions that modify the prompt !!!
Known conflicts: Dynamic Prompts

Appends a "learned" prompt to the end of your prompt that is optimized to maximize the similarity between the text and image embeddings at the end of the diffusion process.

This is done by masking out words in the prompt that are below a threshold given by CLIP e.g. semantic guidance from the paper.

This is useful as is because it allows you to generate images that are (maybe) more similar to the prompt.

The other methods in the paper are not implemented yet. 

I'm not sure how to implement the other methods in the paper because the details of how exactly the "prompt is learned" aren't clear to me. Any insights would be appreciated.

"""


"""

An unofficial and incomplete implementation of Seek for Incantations: Towards Accurate Text-to-Image Diffusion Synthesis
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

class Interrogator:
        def __init__(self):
                pass
        def load(self):
                pass
        def generate_caption(self):
                pass
        def unload(self):
                pass

class InterrogatorCLIP(Interrogator):
        def __init__(self):
                self.interrogator = shared.interrogator

        def load(self):
                self.interrogator = shared.interrogator
                self.interrogator.load()

        def generate_caption(self, pil_image):
                self.load()
                return self.interrogator.generate_caption(pil_image)

        def unload(self):
                self.interrogator.unload()


class InterrogatorDeepbooru(Interrogator):
        def __init__(self):
                self.interrogator = deepbooru.model

        def load(self):
                self.interrogator = deepbooru.model
                self.interrogator.load()

        def generate_caption(self, pil_image):
                self.load()
                if not shared.opts.interrogate_return_ranks: 
                        print('\nincantations - warning: interrogate_return_ranks should be enabled for Deepbooru Interrogate to work')
                threshold = shared.opts.interrogate_deepbooru_score_threshold
                if threshold < 0.4:
                        print('\nincantations - warning: deepbooru score threshold should be lowered for Deepbooru Interrogate to work')
                tags = self.interrogator.tag(pil_image)
                #prompts = prompt_parser.parse_prompt_attention(tags)
                return tags

        def unload(self):
                pass

class IncantStateParams:
        def __init__(self):
                self.delim = ''
                self.word = '-'
                self.coarse = 10
                self.fine = 30
                self.gamma = 0.25
                self.quality = False
                self.deepbooru = False
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
                self.text_tokens = []
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
        
class IncantExtensionScript(UIWrapper):
        def __init__(self):
                self.stage_1 = [[]]
                self.cached_c = [[None, None],[None, None]]
                self.infotext_fields = {}
                self.paste_field_names = []

        # Extension title in menu UI
        def title(self):
                return "Seek Incantations"

        # Setup menu ui detail
        def setup_ui(self, is_img2img):
                return self.setup_seek_incantations()
        
        def get_infotext_fields(self):
                return self.infotext_fields

        def get_paste_field_names(self):
                return self.paste_field_names

        def setup_seek_incantations(self):
                with gr.Accordion('Seek for Incantations', open=True):
                        inc_active = gr.Checkbox(value=False, default=False, label="Active", elem_id='incant_active')
                        inc_quality = gr.Checkbox(value=False, default=False, label="Append Generated Caption", elem_id='incant_append_prompt', info="Append interrogated caption to prompt. (Deepbooru is reversed, if disabled, will not append the masked original prompt)")
                        inc_deepbooru = gr.Checkbox(value=False, default=False, label="Deepbooru Interrogate", elem_id='incant_deepbooru')
                        with gr.Row():
                                inc_delim = gr.Textbox(value='BREAK', label="Delimiter", elem_id='incant_delim', info="Prompt DELIM Optimized Prompt. Try BREAK, AND, NOT, etc.")
                                inc_word = gr.Textbox(value='-', label="Word Replacement", elem_id='incant_word', info="Replace masked words with this")
                        with gr.Row():
                                inc_gamma = gr.Slider(value = 0.2, minimum = -1.0, maximum = 1.0, step = 0.0001, label="Gamma", elem_id = 'incant_gamma', info="If gamma > 0, mask words with similarity less than gamma percent. If gamma < 0, mask more similar words. For Deepbooru, try higher values > 0.7")
                inc_active.do_not_save_to_config = True
                inc_quality.do_not_save_to_config = True
                inc_delim.do_not_save_to_config = True
                inc_word.do_not_save_to_config = True
                inc_deepbooru.do_not_save_to_config = True
                inc_gamma.do_not_save_to_config = True
                self.infotext_fields = [
                        (inc_active, lambda d: gr.Checkbox.update(value='INCANT Active' in d)),
                        (inc_quality, 'INCANT Append Prompt'),
                        (inc_deepbooru, 'INCANT Deepbooru'),
                        (inc_delim, 'INCANT Delim'),
                        (inc_word, 'INCANT Word'),
                        (inc_gamma, 'INCANT Gamma'),
                ]
                self.paste_field_names = [
                        'incant_active',
                        'incant_append_prompt',
                        'incant_deepbooru',
                        'incant_delim',
                        'incant_word',
                        'incant_gamma',
                ]
                return [inc_active, inc_quality, inc_deepbooru, inc_delim, inc_word, inc_gamma]

        def interrogator(self, deepbooru=False): 
                if deepbooru:
                        return InterrogatorDeepbooru()
                else:
                        return shared.interrogator
        
        def before_process(self, p: StableDiffusionProcessing, *args, **kwargs):
                self.incant_before_process(p, *args, **kwargs)

        def incant_before_process(self, p: StableDiffusionProcessing, inc_active, inc_quality, inc_deepbooru, inc_delim, inc_word, inc_gamma, *args, **kwargs):
                inc_active = getattr(p, "incant_active", inc_active)
                if inc_active is False:
                        return
                p.n_iter = p.n_iter * 2
        
        def process(self, p: StableDiffusionProcessing, *args, **kwargs):
                self.incant_process(p, *args, **kwargs)

        def incant_process(self, p: StableDiffusionProcessing, inc_active, inc_quality, inc_deepbooru, inc_delim, inc_word, inc_gamma, *args, **kwargs):
                inc_active = getattr(p, "incant_active", inc_active)
                if inc_active is False:
                        return
                inc_quality = getattr(p, "incant_append_prompt", inc_quality)
                inc_delim = getattr(p, "incant_delim", inc_delim)
                inc_word = getattr(p, "incant_word", inc_word)
                        
                        # modifying the all_prompts* may conflict with extensions that do so
                        # hr fix untested
                if p.iteration == 0:
                        param_list = [
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
                        delim_str = f' {inc_delim} ' if len(inc_delim) > 0 else ' '
                        for n in range(1, p.n_iter, 2):
                                start_idx = n * p.batch_size
                                end_idx = (n + 1) * p.batch_size
                                p.all_prompts[start_idx:end_idx] = [prompt + delim_str + '<<REPLACEME>>' for prompt in p.all_prompts[start_idx:end_idx]]

        def before_process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
                self.incant_before_process_batch(p, *args, **kwargs)

        def incant_before_process_batch(self, p: StableDiffusionProcessing, inc_active, inc_quality, inc_deepbooru, inc_delim, inc_word, inc_gamma, *args, **kwargs):
                inc_active = getattr(p, "incant_active", inc_active)
                if inc_active is False:
                        return
                inc_quality = getattr(p, "incant_append_prompt", inc_quality)
                inc_deepbooru = getattr(p, "incant_deepbooru", inc_deepbooru)
                inc_delim = getattr(p, "incant_delim", inc_delim)
                inc_word = getattr(p, "incant_word", inc_word)
                fine_step = getattr(p, "incant_fine", None)
                inc_gamma = getattr(p, "incant_gamma", inc_gamma)
                if fine_step == None:
                        #print(f"Fine step {fine_step} is greater than total steps {p.steps}, setting to {p.steps-2}")
                        fine_step = p.steps - 2

                interrogator = self.interrogator(inc_deepbooru)
                interrogator.load()
                n = p.n_iter
                if p.iteration % 2 == 1:
                        n = p.iteration
                                # batch of images
                        batch_start_idx = n * p.batch_size
                        batch_end_idx = (n + 1) * p.batch_size
                                # mask 
                        mask_start_idx = (n - 1) * p.batch_size
                        delim_str = f' {inc_delim} ' if len(inc_delim) > 0 else ' '
                                #add_mask_prompt = self.stage_1.masked_prompt[mask_start_idx]
                        for idx in range(batch_start_idx, batch_end_idx):
                                add_mask_prompt = ''
                                mask_idx = mask_start_idx + (idx - batch_start_idx)
                                masked_prompts = self.stage_1.masked_prompt[mask_idx]
                                        # if we don't want to append other masked captions, only use the first one
                                if not inc_quality:
                                        masked_prompts = [masked_prompts[0]]
                                for masked_prompt_idx, prompt in enumerate(masked_prompts):
                                        if masked_prompt_idx > 0:
                                                add_mask_prompt += delim_str + prompt 
                                        else:
                                                add_mask_prompt += prompt
                                p.all_prompts[idx] = p.all_prompts[idx].replace('<<REPLACEME>>', add_mask_prompt)
                                kwargs['prompts'][mask_idx] = kwargs['prompts'][mask_idx].replace('<<REPLACEME>>', add_mask_prompt)

                        # p.steps += fine_step
                        # TODO: nicely put this into a dict
                p.extra_generation_params.update({
                                "INCANT Active": inc_active,
                                "INCANT Append Prompt": inc_quality,
                                "INCANT Delim": inc_delim,
                                "INCANT Word": inc_word,
                                "INCANT Deepbooru": inc_deepbooru,
                                "INCANT Fine": fine_step,
                                "INCANT Gamma": inc_gamma,
                })
                self.create_hook(p, inc_active, inc_quality, inc_deepbooru, inc_delim, inc_word, inc_gamma, *args, **kwargs)
        
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

        def create_hook(self, p, active, quality, deepbooru, delim, word, gamma, *args, **kwargs):

                import clip
                # Create a list of parameters for each concept
                incant_params = IncantStateParams()
                incant_params.p = p
                incant_params.prompt = p.prompt
                incant_params.prompts = [pr for pr in p.prompts]
                #incant_params.prompt_tokens = clip.tokenize(list(p.prompt), truncate=True).to(devices.device_interrogate)
                incant_params.quality = quality 
                #incant_params.coarse = coarse
                incant_params.delim = delim
                incant_params.word = word 
                incant_params.deepbooru = deepbooru
                fine = p.steps
                incant_params.fine = p.steps
                if incant_params.fine >= p.steps:
                        print(f"Fine step {fine} is greater than total steps {p.steps}, setting to {p.steps}")
                        fine = max(p.steps - 2, 1)
                incant_params.gamma = gamma
                incant_params.qual_scale = 0
                incant_params.sem_scale = 0
                incant_params.iteration = p.iteration
                incant_params.get_conds_with_caching = p.get_conds_with_caching
                incant_params.steps = p.steps
                incant_params.batch_size = p.batch_size
                incant_params.job = shared.state.job
                #incant_params.first_stage_cache = self.stage_1[0]
                incant_params.second_stage = (p.iteration % 2) == 1
                tqdm = shared.total_tqdm
                if not hasattr(p, 'incant_params'):
                        setattr(p, 'incant_params', incant_params)

                if p.iteration % 2 == 0:
                        self.stage_1 = incant_params
                else:
                        # assign old cache to next iteration
                        incant_params.first_stage_cache = self.stage_1

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, incant_params)
                y2 = lambda params: self.cfg_after_cfg_callback(params, incant_params)
                #y3 = lambda params: self.cfg_after_cfg_callback(params, incant_params)

                logger.debug('Hooked callbacks')
                #script_callbacks.on_cfg_denoised(y2)
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_cfg_after_cfg(y2)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)
        
        def calc_masked_prompt(self, incant_params: IncantStateParams, first_stage_cache):
                fs = first_stage_cache
                repl_word = '-'
                prompt = incant_params.p.prompt
                repl_threshold = incant_params.gamma * 100.0
                word_list = fs.matches_fine[0]
                masked_prompt = self.mask_prompt(repl_threshold, word_list, incant_params.p.prompt, incant_params.word)
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
        
        def postprocess_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
                return self.incant_postprocess_batch(p, *args, **kwargs)

        def incant_postprocess_batch(self, p: StableDiffusionProcessing, inc_active, *args, **kwargs):
                inc_active = getattr(p, "incant_active", inc_active)
                if inc_active is False:
                        return
                batch_number = kwargs.get('batch_number', -1)
                images = kwargs.get('images', None)
                incant_params: IncantStateParams = getattr(p, "incant_params", None)
                to_pil = ToPILImage()

                n = p.iteration
                if n % 2 == 0:
                        #fine_images = self.decode_images(images)
                        fine_images = [to_pil(img) for img in images]
                        incant_params.img_fine.extend(fine_images)
                        self.interrogate_images(incant_params, p)
                        devices.torch_gc()
                        # compute masked_prompts
                        batch_start_idx = n * p.batch_size
                        batch_end_idx = (n + 1) * p.batch_size
                        for batch_idx, caption_matches_item in enumerate(incant_params.matches_fine[batch_start_idx:batch_end_idx]):
                                batch_mask_prompts = []
                                for caption, matches in caption_matches_item:
                                        batch_mask_prompts.append(self.mask_prompt(incant_params.gamma, matches, caption, incant_params.word))
                                incant_params.masked_prompt.append(batch_mask_prompts)

                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                interrogator = self.interrogator(False)
                interrogator.unload()
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

                # temp bypass
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

        def mask_prompt(self, gamma, word_list, prompt, word_repl = '-'):
                # TODO: refactor out removing <>
                regex = r"\b{0}\b"
                masked_prompt = prompt
                mask_less_similar = gamma > 0
                gamma = abs(gamma)
                for word, pct in word_list: 
                        word = word.strip(', ')
                        if len(word) == 0:
                                continue
                        condition = (pct < gamma) if mask_less_similar else (pct > gamma)
                        if condition:
                                repl_regex = regex.format(word)
                                        # replace word with -
                                masked_prompt = re.sub(repl_regex, word_repl, masked_prompt)
                        is_lora = word.startswith('<') and word.endswith('>')
                        if is_lora:
                                masked_prompt = masked_prompt.replace(word, '')
                # hack: remove text between pairs of brackets like <...>
                masked_prompt = re.sub(r'<[^>]*>', '', masked_prompt)

                return masked_prompt

        def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams, incant_params: IncantStateParams):
                pass
                #p = incant_params.p
                #fine = incant_params.fine
                #second_stage = incant_params.second_stage
                #x = params.x

                ## BUG: webui params passes shared.state.sampling_step instead of the internal self.step
                #if shared.state is not None:
                #        step = max(params.sampling_step, shared.state.sampling_step)
                #else:
                #        step = params.sampling_step

                ## FIXME: why is the max value of step 2 less than the total steps???
                #if step == fine - 2 and not second_stage:
                #        # print(f"\nFine step: {step}\n")

                #        # decode fine images
                #        fine_images = self.decode_images(x)
                #        incant_params.img_fine = fine_images 

                #        # compute embedding stuff
                #        self.interrogate_images(incant_params, p)
                #        devices.torch_gc()

                #        # compute masked_prompts
                #        for batch_idx, caption_matches_item in enumerate(incant_params.matches_fine):
                #                batch_mask_prompts = []
                #                for caption, matches in caption_matches_item:
                #                        batch_mask_prompts.append(self.mask_prompt(incant_params.gamma, matches, caption, incant_params.word))
                #                incant_params.masked_prompt.append(batch_mask_prompts)
                #else:
                #        pass

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
                interrogator = self.interrogator(incant_params.deepbooru)
                interrogator.load()

                # fine features
                for batch_idx, pil_image in enumerate(incant_params.img_fine):
                        # generate caption
                        caption = interrogator.generate_caption(pil_image)
                        devices.torch_gc()

                        # CLIP
                        if not incant_params.deepbooru:
                                matches_list = []
                                # append caption
                                caption = interrogator.generate_caption(pil_image)
                                incant_params.caption_fine.append(caption)

                                # calculate image embeddings
                                image_features = self.calc_img_embedding(interrogator, pil_image)
                                incant_params.emb_img_fine.append(image_features)

                                # calculate image similarity to prompt
                                prompt_text_array = incant_params.prompt.split()
                                matches = self.clip_text_image_similarity(interrogator, prompt_text_array, image_features, top_count=len(prompt_text_array))
                                matches_list.append((incant_params.prompt, matches))
                                #print(f"\n{batch_idx}-prompt:{caption}\n{batch_idx}-fine:{matches}\n")

                                # calculate image similarity to generated caption
                                if incant_params.quality:
                                        caption_text_array = caption.split()
                                        matches = interrogator.rank(image_features, caption_text_array, top_count=len(caption_text_array))
                                        matches_list.append((caption, matches))
                                        print(f"\n{batch_idx}-fine:{matches}\n")

                                incant_params.matches_fine.append(matches_list)

                        # deepbooru interrogate
                        else:
                                matches_list = []
                                # TODO: separate options to append generated caption and append masked original prompt
                                # for deepbooru, if disabled, append generated caption will not append the ORIGINAL prompt
                                # mask the original prompt
                                if incant_params.quality:
                                        new_prompt, prompt_matches_list = self.interrogate_deepbooru(incant_params.prompt, incant_params.gamma)
                                        matches_list.append((new_prompt, prompt_matches_list))
                                        print(f"{batch_idx}-prompt:{new_prompt}\n")

                                new_caption, caption_matches_list = self.interrogate_deepbooru(caption, incant_params.gamma)
                                matches_list.append((new_caption, caption_matches_list))
                                print(f"{batch_idx}-caption:{new_caption}\n")

                                incant_params.caption_fine.append(new_caption)
                                #incant_params.caption_fine.append(new_prompt)

                                # append auto generated captions
                                incant_params.matches_fine.append(matches_list)

                devices.torch_gc()

        def interrogate_deepbooru(self, caption, gamma):
                """_summary_

                Args:
                    caption (_type_): _description_
                    matches_list (_type_): _description_
                    gamma (_type_): _description_
                    mask_less_similar (_type_): _description_

                Returns:
                    _type_: _description_
                """

                mask_less_similar = gamma > 0
                gamma = abs(gamma)

                matches_list = []
                # preprocess caption

                # remove lora
                caption = re.sub(r'<[^>]*>', '', caption)

                matches = prompt_parser.parse_prompt_attention(caption)
                if mask_less_similar:
                        matches = [(tag, strength) for (tag, strength) in matches if strength >= gamma]
                else:
                        matches = [(tag, strength) for (tag, strength) in matches if strength < gamma]
                                        # matches = [(tag, strength) for (tag, strength) in matches]
                                        # split by tags
                for tags, strength in matches:
                        for tag in tags.split(', '):
                                if len(tag) == 0:
                                        continue
                                                        # filter tags
                                matches_list.append((tag.strip(), strength))

                new_caption = ''
                for tag, strength in matches_list:
                        new_caption += f'({tag}:{strength}), '
                new_caption.removesuffix(', ')
                return new_caption, matches_list

        def clip_text_image_similarity(self, interrogator, text_array, image_features, top_count=1) -> list[tuple[str, float]]:
                """ Calculate similarity between text and image features using CLIP

                Args:
                    interrogator (): shared.interrogator
                    text_array (str): text to match similarity
                    image_features (tensor): image encoded with calc_img_embedding
                    top_count (int, optional): number of top matches to return

                Returns:
                    list[tuple[str, float]]: _description_
                """
                with torch.no_grad(), devices.autocast():
                        matches = interrogator.rank(image_features, text_array, top_count=top_count)
                        matches = [(tag, strength/100.0) for (tag, strength) in matches] # rescale to 0-1
                return matches

        def calc_img_embedding(self, interrogator, pil_image):
                clip_image = interrogator.clip_preprocess(pil_image).unsqueeze(0).type(interrogator.dtype).to(devices.device_interrogate)
                with torch.no_grad(), devices.autocast():
                                                # calculate image embeddings
                        image_features = interrogator.clip_model.encode_image(clip_image).type(interrogator.dtype)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                return image_features

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

        def get_xyz_axis_options(self) -> dict:
                xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ == "xyz_grid.py"][0].module
                extra_axis_options = {
                        xyz_grid.AxisOption("[Incant] Active", str, incant_apply_override('incant_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[Incant] Append Caption Prompt", str, incant_apply_override('incant_append_prompt', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[Incant] Deepbooru Interrogate", str, incant_apply_override('incant_deepbooru', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                        xyz_grid.AxisOption("[Incant] Delimiter", str, incant_apply_field("incant_delim")),
                        xyz_grid.AxisOption("[Incant] Replacement Word", str, incant_apply_field("incant_word")),
                        xyz_grid.AxisOption("[Incant] Gamma", float, incant_apply_field("incant_gamma"))
                }
                return extra_axis_options


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
# untested
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

