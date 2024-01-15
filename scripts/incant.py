import logging
from os import environ
import modules.scripts as scripts
import gradio as gr
import scipy.stats as stats

from modules import script_callbacks, prompt_parser
from modules.script_callbacks import CFGDenoiserParams
from modules.prompt_parser import reconstruct_multicond_batch
from modules.processing import StableDiffusionProcessing
#from modules.shared import sd_model, opts
from modules.sd_samplers_cfg_denoiser import pad_cond
from modules import shared

import torch

logger = logging.getLogger(__name__)
logger.setLevel(environ.get("SD_WEBUI_LOG_LEVEL", logging.INFO))

"""

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
                self.concept_name = ''
                self.v = {} # velocity
                self.warmup_period: int = 10 # [0, 20]
                self.edit_guidance_scale: float = 1 # [0., 1.]
                self.tail_percentage_threshold: float = 0.05 # [0., 1.] if abs value of difference between uncodition and concept-conditioned is less than this, then zero out the concept-conditioned values less than this
                self.momentum_scale: float = 0.3 # [0., 1.]
                self.momentum_beta: float = 0.6 # [0., 1.) # larger bm is less volatile changes in momentum
                self.strength = 1.0

class IncantExtensionScript(scripts.Script):
        def __init__(self):
                self.cached_c = [None, None]

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
                        with gr.Row():
                                coarse_step = gr.Slider(value = 10, minimum = 0, maximum = 100, step = 1, label="Coarse Step", elem_id = 'incant_coarse')
                                fine_step = gr.Slider(value = 10, minimum = 0, maximum = 100, step = 1, label="Fine Step", elem_id = 'incant_fine')
                        # with gr.Row():
                        #         warmup = gr.Slider(value = 10, minimum = 0, maximum = 30, step = 1, label="Warmup Period", elem_id = 'incant_warmup', info="How many steps to wait before applying semantic guidance, default 10")
                        #         edit_guidance_scale = gr.Slider(value = 1.0, minimum = 0.0, maximum = 20.0, step = 0.01, label="Edit Guidance Scale", elem_id = 'incant_edit_guidance_scale', info="Scale of edit guidance, default 1.0")
                        #         tail_percentage_threshold = gr.Slider(value = 0.05, minimum = 0.0, maximum = 1.0, step = 0.01, label="Tail Percentage Threshold", elem_id = 'incant_tail_percentage_threshold', info="The percentage of latents to modify, default 0.05")
                        #         momentum_scale = gr.Slider(value = 0.3, minimum = 0.0, maximum = 1.0, step = 0.01, label="Momentum Scale", elem_id = 'incant_momentum_scale', info="Scale of momentum, default 0.3")
                        #         momentum_beta = gr.Slider(value = 0.6, minimum = 0.0, maximum = 0.999, step = 0.01, label="Momentum Beta", elem_id = 'incant_momentum_beta', info="Beta for momentum, default 0.6")
                active.do_not_save_to_config = True
                coarse_step.do_not_save_to_config = True
                fine_step.do_not_save_to_config = True
                # warmup.do_not_save_to_config = True
                # edit_guidance_scale.do_not_save_to_config = True
                # tail_percentage_threshold.do_not_save_to_config = True
                # momentum_scale.do_not_save_to_config = True
                # momentum_beta.do_not_save_to_config = True
                # self.infotext_fields = [
                #         (active, lambda d: gr.Checkbox.update(value='INCANT Active' in d)),
                #         (coarse_step, 'INCANT Prompt'),
                #         (fine_step, 'INCANT Negative Prompt'),
                #         (warmup, 'INCANT Warmup Period'),
                #         (edit_guidance_scale, 'INCANT Edit Guidance Scale'),
                #         (tail_percentage_threshold, 'INCANT Tail Percentage Threshold'),
                #         (momentum_scale, 'INCANT Momentum Scale'),
                #         (momentum_beta, 'INCANT Momentum Beta'),
                # ]
                # self.paste_field_names = [
                #         'incant_active',
                #         'incant_prompt',
                #         'incant_neg_prompt',
                #         'incant_warmup',
                #         'incant_edit_guidance_scale',
                #         'incant_tail_percentage_threshold',
                #         'incant_momentum_scale',
                #         'incant_momentum_beta'
                # ]
                return [active, coarse_step, fine_step]

        def process_batch(self, p: StableDiffusionProcessing, active, coarse_step, fine_step, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                coarse_step = getattr(p, "incant_coarse", coarse_step)
                fine_step = getattr(p, "incant_fine", fine_step)

                p.extra_generation_params = {
                        "INCANT Active": active,
                        "INCANT Coarse": coarse_step,
                        "INCANT Fine": fine_step,
                }

                # separate concepts by comma
                # concept_prompts = self.parse_concept_prompt(prompt)
                # concept_prompts_neg = self.parse_concept_prompt(neg_prompt)
                # # [[concept_1,  strength_1], ...]
                # concept_prompts = [prompt_parser.parse_prompt_attention(concept)[0] for concept in concept_prompts]
                # concept_prompts_neg = [prompt_parser.parse_prompt_attention(neg_concept)[0] for neg_concept in concept_prompts_neg]
                # concept_prompts_neg = [[concept, -strength] for concept, strength in concept_prompts_neg]
                # concept_prompts.extend(concept_prompts_neg)

                # concept_conds = []
                # for concept, strength in concept_prompts:
                #         prompt_list = [concept] * p.batch_size
                #         prompts = prompt_parser.SdConditioning(prompt_list, width=p.width, height=p.height)
                #         c = p.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, [self.cached_c], p.extra_network_data)
                #         concept_conds.append([c, strength])

                # self.create_hook(p, active, concept_conds, None, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta)

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

        def create_hook(self, p, active, concept_conds, concept_conds_neg, warmup, edit_guidance_scale, tail_percentage_threshold, momentum_scale, momentum_beta, *args, **kwargs):
                # Create a list of parameters for each concept
                concepts_incant_params = []
                for _, strength in concept_conds:
                        incant_params = IncantStateParams()
                        incant_params.warmup_period = warmup
                        incant_params.edit_guidance_scale = edit_guidance_scale
                        incant_params.tail_percentage_threshold = tail_percentage_threshold
                        incant_params.momentum_scale = momentum_scale
                        incant_params.momentum_beta = momentum_beta
                        incant_params.strength = strength
                        concepts_incant_params.append(incant_params)

                # Use lambda to call the callback function with the parameters to avoid global variables
                y = lambda params: self.on_cfg_denoiser_callback(params, concept_conds, concepts_incant_params)

                logger.debug('Hooked callbacks')
                script_callbacks.on_cfg_denoiser(y)
                script_callbacks.on_script_unloaded(self.unhook_callbacks)

        def postprocess_batch(self, p, active, neg_text, *args, **kwargs):
                active = getattr(p, "incant_active", active)
                if active is False:
                        return
                self.unhook_callbacks()

        def unhook_callbacks(self):
                logger.debug('Unhooked callbacks')
                script_callbacks.remove_current_script_callbacks()

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, concept_conds, incant_params: list[IncantStateParams]):
                # TODO: add option to opt out of batching for performance
                sampling_step = params.sampling_step
                text_cond = params.text_cond
                text_uncond = params.text_uncond

                # pad text_cond or text_uncond to match the length of the longest prompt
                # i would prefer to let sd_samplers_cfg_denoiser.py handle the padding, but
                # there isn't a callback that returns the padded conds
                if text_cond.shape[1] != text_uncond.shape[1]:
                        empty = shared.sd_model.cond_stage_model_empty_prompt
                        num_repeats = (text_cond.shape[1] - text_uncond.shape[1]) // empty.shape[1]

                        if num_repeats < 0:
                                text_cond = pad_cond(text_cond, -num_repeats, empty)
                        elif num_repeats > 0:
                                text_uncond = pad_cond(text_uncond, num_repeats, empty)

                batch_conds_list = []
                batch_tensor = {}

                # sd 1.5 support
                if isinstance(text_cond, torch.Tensor):
                        text_cond = {'crossattn': text_cond}
                if isinstance(text_uncond, torch.Tensor):
                        text_uncond = {'crossattn': text_uncond}

                for i, _ in enumerate(incant_params):
                        concept_cond, _ = concept_conds[i]
                        conds_list, tensor_dict = reconstruct_multicond_batch(concept_cond, sampling_step)

                        # sd 1.5 support
                        if isinstance(tensor_dict, torch.Tensor):
                                tensor_dict = {'crossattn': tensor_dict}

                        # initialize here because we don't know the shape/dtype of the tensor until we reconstruct it
                        for key, tensor in tensor_dict.items():
                                if tensor.shape[1] != text_uncond[key].shape[1]:
                                        empty = shared.sd_model.cond_stage_model_empty_prompt
                                        num_repeats = (tensor.shape[1] - text_uncond.shape[1]) // empty.shape[1]
                                        if num_repeats < 0:
                                                tensor = pad_cond(tensor, -num_repeats, empty)
                                tensor = tensor.unsqueeze(0)
                                if key not in batch_tensor.keys():
                                        batch_tensor[key] = tensor
                                else:
                                        batch_tensor[key] = torch.cat((batch_tensor[key], tensor), dim=0)
                        batch_conds_list.append(conds_list)
                self.incant_routine_batch(params, batch_conds_list, batch_tensor, incant_params, text_cond, text_uncond)
        
        def make_tuple_dim(self, dim):
                # sd 1.5 support
                if isinstance(dim, torch.Tensor):
                        dim = dim.dim()
                return (-1,) + (1,) * (dim - 1)

        def incant_routine_batch(self, params: CFGDenoiserParams, batch_conds_list, batch_tensor, incant_params: list[IncantStateParams], text_cond, text_uncond):
                # FIXME: these parameters should be specific to each concept
                warmup_period = incant_params[0].warmup_period
                edit_guidance_scale = incant_params[0].edit_guidance_scale
                tail_percentage_threshold = incant_params[0].tail_percentage_threshold
                momentum_scale = incant_params[0].momentum_scale
                momentum_beta = incant_params[0].momentum_beta

                sampling_step = params.sampling_step

                # Semantic Guidance
                edit_dir_dict = {}

                # batch_tensor: [num_concepts, batch_size, tokens(77, 154, etc.), 2048]
                # Calculate edit direction
                for key, concept_cond in batch_tensor.items():
                        new_shape = self.make_tuple_dim(concept_cond)
                        strength = torch.Tensor([params.strength for params in incant_params]).to(dtype=concept_cond.dtype, device=concept_cond.device)
                        strength = strength.view(new_shape)

                        if key not in edit_dir_dict.keys():
                                edit_dir_dict[key] = torch.zeros_like(concept_cond, dtype=concept_cond.dtype, device=concept_cond.device)

                        # filter out values in-between tails
                        # FIXME: does this take into account image batch size?, i.e. dim 1
                        inside_dim = tuple(range(-concept_cond.dim() + 1, 0)) # for tensor of dim 4, returns (-3, -2, -1), for tensor of dim 3, returns (-2, -1)
                        cond_mean, cond_std = torch.mean(concept_cond, dim=inside_dim), torch.std(concept_cond, dim=inside_dim)

                        # broadcast element-wise subtraction
                        edit_dir = concept_cond - text_uncond[key]

                        # multiply by strength for positive / negative direction
                        edit_dir = torch.mul(strength, edit_dir)

                        # z-scores for tails
                        upper_z = stats.norm.ppf(1.0 - tail_percentage_threshold)

                        # numerical thresholds
                        # FIXME: does this take into account image batch size?, i.e. dim 1
                        upper_threshold = cond_mean + (upper_z * cond_std)

                        # reshape to be able to broadcast / use torch.where to filter out values for each concept
                        #new_shape = (-1,) + (1,) * (concept_cond.dim() - 1)
                        new_shape = self.make_tuple_dim(concept_cond)
                        upper_threshold_reshaped = upper_threshold.view(new_shape)

                        # zero out values in-between tails
                        # elementwise multiplication between scale tensor and edit direction
                        zero_tensor = torch.zeros_like(concept_cond, dtype=concept_cond.dtype, device=concept_cond.device)
                        scale_tensor = torch.ones_like(concept_cond, dtype=concept_cond.dtype, device=concept_cond.device) * edit_guidance_scale
                        edit_dir_abs = edit_dir.abs()
                        scale_tensor = torch.where((edit_dir_abs > upper_threshold_reshaped), scale_tensor, zero_tensor)

                        # update edit direction with the edit dir for this concept
                        guidance_strength = 0.0 if sampling_step < warmup_period else 1.0 # FIXME: Use appropriate guidance strength
                        edit_dir = torch.mul(scale_tensor, edit_dir)
                        edit_dir_dict[key] = edit_dir_dict[key] + guidance_strength * edit_dir

                # TODO: batch this
                for i, incant_param in enumerate(incant_params):
                        for key, dir in edit_dir_dict.items():
                                # calculate momentum scale and velocity
                                if key not in incant_param.v.keys():
                                        slice_idx = 1 - dir.dim() # should be negative, for dim=4, slice_idx = -3
                                        incant_param.v[key] = torch.zeros(dir.shape[slice_idx:], dtype=dir.dtype, device=dir.device)

                                # add to text condition
                                v_t = incant_param.v[key]
                                dir[i] = dir[i] + torch.mul(momentum_scale, v_t)

                                # calculate v_t+1 and update state
                                v_t_1 = momentum_beta * ((1 - momentum_beta) * v_t) * dir[i]

                                # add to cond after warmup elapsed
                                # for sd 1.5, we must add to the original params.text_cond because we reassigned text_cond
                                if sampling_step >= warmup_period:
                                        if isinstance(params.text_cond, dict):
                                                params.text_cond[key] = params.text_cond[key] + dir[i]
                                        else:
                                                params.text_cond = params.text_cond + dir[i]

                                # update velocity
                                incant_param.v[key] = v_t_1

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
