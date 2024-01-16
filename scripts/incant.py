import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
from PIL import Image
import numpy as np
import re

from modules import script_callbacks, prompt_parser
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
from modules.prompt_parser import reconstruct_multicond_batch, stack_conds
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
                self.matches_coarse = []
                self.matches_fine = []
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

        def before_process(self, p: StableDiffusionProcessing, active, quality, coarse, fine, gamma, *args, **kwargs):
                p.n_iter = p.n_iter * 2

        # Setup menu ui detail
        def ui(self, is_img2img):
                with gr.Accordion('Incantations', open=False):
                        active = gr.Checkbox(value=False, default=False, label="Active", elem_id='incant_active')
                        quality = gr.Checkbox(value=True, default=True, label="Quality Guidance", elem_id='incant_quality')
                        with gr.Row():
                                coarse_step = gr.Slider(value = 10, minimum = 0, maximum = 100, step = 1, label="Coarse Step", elem_id = 'incant_coarse')
                                fine_step = gr.Slider(value = 30, minimum = 0, maximum = 100, step = 1, label="Fine Step", elem_id = 'incant_fine')
                                gamma = gr.Slider(value = 0.25, minimum = 0, maximum = 1.0, step = 0.01, label="Gamma", elem_id = 'incant_gamma')
                active.do_not_save_to_config = True
                quality.do_not_save_to_config = True
                coarse_step.do_not_save_to_config = True
                fine_step.do_not_save_to_config = True
                gamma.do_not_save_to_config = True
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
                return [active, quality, coarse_step, fine_step, gamma]

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

        def before_process_batch(self, p: StableDiffusionProcessing, active, quality, coarse_step, fine_step, gamma, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                quality = getattr(p, "incant_quality", quality)
                coarse_step = getattr(p, "incant_coarse", coarse_step)
                fine_step = getattr(p, "incant_fine", fine_step)
                gamma = getattr(p, "incant_gamma", gamma)
                # if fine_step > p.steps:
                #         print(f"Fine step {fine_step} is greater than total steps {p.steps}, setting to {p.steps}")
                #         fine_step = p.steps

                interrogator = shared.interrogator
                interrogator.load()
                # Every even step will be when we use the previously calculated results
                # p.n_iter = p.n_iter * 2
                print(f"n_iter: {p.n_iter}")

                # Duplicate every element in each list if it exists
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
                        run_fn_on_attr(p, param_name, duplicate_alternate_elements)
                # p.steps += fine_step

                p.extra_generation_params = {
                        "INCANT Active": active,
                        "INCANT Quality": quality,
                        "INCANT Coarse": coarse_step,
                        "INCANT Fine": fine_step,
                        "INCANT Gamma": gamma,
                }

                self.create_hook(p, active, quality, coarse_step, fine_step, gamma, *args, **kwargs)

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

        def create_hook(self, p, active, quality, coarse, fine, gamma, *args, **kwargs):
                import clip
                # Create a list of parameters for each concept
                incant_params = IncantStateParams()
                incant_params.p = p
                incant_params.prompt = p.prompt
                incant_params.prompts = [pr for pr in p.prompts]
                #incant_params.prompt_tokens = clip.tokenize(list(p.prompt), truncate=True).to(devices.device_interrogate)
                incant_params.coarse = coarse
                incant_params.fine = fine 
                incant_params.gamma = gamma
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
                        self.calc_quality_guidance(incant_params, self.stage_1)

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, incant_params)
                y2 = lambda params: self.on_cfg_denoised_callback(params, incant_params)
                y3 = lambda params: self.cfg_after_cfg_callback(params, incant_params)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_cfg_denoised(y2)
                script_callbacks.cfg_after_cfg_callback(y3)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)
        
        def calc_masked_prompt(self, incant_params: IncantStateParams, first_stage_cache):
                fs = first_stage_cache
                repl_word = '-'
                prompt = incant_params.p.prompt
                repl_threshold = incant_params.gamma * 100.0
                word_list = fs.matches_fine[0]
                masked_prompt = self.mask_prompt(repl_threshold, word_list, incant_params.p.prompt)
                return masked_prompt

        def calc_quality_guidance(self, incant_params: IncantStateParams, first_stage_cache):


                # mask prompt
                pass
        
        def cfg_after_cfg_callback(self, params: AfterCFGCallbackParams, incant_params):
                #active = getattr(params, "incant_active", active)
                pass


        def postprocess_batch(self, p, active, quality, coarse, fine, gamma, *args, **kwargs):
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

                fs = incant_params.first_stage_cache

                # TODO: handle batches

                p = incant_params.p
                sampling_step = params.sampling_step
                text_cond = params.text_cond
                text_uncond = params.text_uncond
                gamma = incant_params.gamma * 100.0

                # generic regex to replace whole words that match
#                 word_list = fs.matches_fine[0]
#                 # compute masked prompt
#                 masked_prompt = p.prompt
#                 similarities = []
#                 mask_prompt = self.mask_prompt(gamma, word_list, masked_prompt)
#                 masked_prompt = mask_prompt
#                 print(f'\nmasked prompt: "{masked_prompt}"\n')
#                 prompt_list = [masked_prompt] * p.batch_size
#                 prompts = prompt_parser.SdConditioning(prompt_list, width=p.width, height=p.height)
#                 c = incant_params.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, self.cached_c, p.extra_network_data)
# 
                # # # concatenate text_cond with c
                # # text_cond = torch.cat((text_cond, c), dim=1)

                # # pad text_cond or text_uncond to match the length of the longest prompt
                # # i would prefer to let sd_samplers_cfg_denoiser.py handle the padding, but
                # # there isn't a callback that returns the padded conds
                # # if text_cond.shape[1] != text_uncond.shape[1]:
                # #         empty = shared.sd_model.cond_stage_model_empty_prompt
                # #         num_repeats = (text_cond.shape[1] - text_uncond.shape[1]) // empty.shape[1]

                # #         if num_repeats < 0:
                # #                 text_cond = pad_cond(text_cond, -num_repeats, empty)
                # #         elif num_repeats > 0:
                # #                 text_uncond = pad_cond(text_uncond, num_repeats, empty)

                # batch_conds_list = []
                # batch_tensor = {}

                # # sd 1.5 support
                # if isinstance(text_cond, torch.Tensor):
                #         text_cond = {'crossattn': text_cond}
                # if isinstance(text_uncond, torch.Tensor):
                #         text_uncond = {'crossattn': text_uncond}

                # # for i, _ in enumerate(sega_params):
                # # concept_cond, _ = concept_conds[i]
                # conds_list, tensor_dict = reconstruct_multicond_batch(c, sampling_step)

                # # sd 1.5 support
                # if isinstance(tensor_dict, torch.Tensor):
                #         tensor_dict = {'crossattn': tensor_dict}

                # # initialize here because we don't know the shape/dtype of the tensor until we reconstruct it
                # for key, tensor in tensor_dict.items():
                #         if tensor.shape[1] != text_uncond[key].shape[1]:
                #                 empty = shared.sd_model.cond_stage_model_empty_prompt
                #                 num_repeats = (tensor.shape[1] - text_uncond.shape[1]) // empty.shape[1]
                #                 if num_repeats < 0:
                #                         tensor = pad_cond(tensor, -num_repeats, empty)
                #         tensor = tensor.unsqueeze(0)
                #         if key not in batch_tensor.keys():
                #                 batch_tensor[key] = tensor
                #         else:
                #                 batch_tensor[key] = torch.cat((batch_tensor[key], tensor), dim=0)
                # batch_conds_list.append(conds_list)
                # if isinstance(params.text_cond, dict):
                #         params.text_cond[key] = params.text_cond[key].cat(batch_tensor[key], dim=1)
                # else:
                #         params.text_cond = stack_conds([params.text_cond, batch_tensor[key].squeeze(0)])
                #         #params.text_cond = params.text_cond.cat(batch_tensor[key], dim=1)
                # if params.text_cond.shape[1] != params.text_uncond.shape[1]:
                #         empty = shared.sd_model.cond_stage_model_empty_prompt
                #         num_repeats = (params.text_cond.shape[1] - params.text_uncond.shape[1]) // empty.shape[1]

                #         if num_repeats < 0:
                #                 params.text_cond = pad_cond(params.text_cond, -num_repeats, empty)
                #         elif num_repeats > 0:
                #                 params.text_uncond = pad_cond(params.text_uncond, num_repeats, empty)

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

        def on_cfg_denoised_callback(self, params: CFGDenoisedParams, incant_params: IncantStateParams):
                import clip
                p = incant_params.p
                coarse = incant_params.coarse
                fine = incant_params.fine
                x = params.x
                step = params.sampling_step
                max_step = params.total_sampling_steps
                second_stage = incant_params.second_stage

                # save the coarse images
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

                        self.interrogate_images(incant_params, p)
                        devices.torch_gc()
                        
                else:
                        pass

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
                                image_features /= image_features.norm(dim=-1, keepdim=True)
                                clip_img_embed_list.append(image_features)

                                # calculate text embeddings
                                prompt_list = [caption] * p.batch_size
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
                                prompt_list = [caption] * p.batch_size
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
            x_samples_ddim = decode_latent_batch(shared.sd_model, x, target_device=devices.cpu, check_for_nans=True)
            x_samples_ddim = torch.stack(x_samples_ddim).float()
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            for i, x_sample in enumerate(x_samples_ddim):
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)
                    image = Image.fromarray(x_sample)
                    batch_images.append(image)
            return batch_images


def run_fn_on_attr(p, attr_name, fn):
        """ Run a function on an attribute of a class if it exists """
        try:
                attr = getattr(p, attr_name)
                setattr(p, attr_name, fn(attr))
        except AttributeError:
                # No attribute exists
                return
        except TypeError:
                # If the attribute is not iterable, return
                return

def duplicate_alternate_elements(input_list: list) -> list:
        """ Duplicate each element in a list and return a new list
        >>> duplicate_list([1,2,3,4])
        [1, 1, 3, 3]
        """
        result = input_list
        for idx in range(0, len(result), 2):
                result[idx+1] = result[idx]
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
