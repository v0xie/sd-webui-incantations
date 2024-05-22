import os
import gradio as gr
from functools import reduce
from scripts.incant_utils import prompt_utils
from scripts.ui_wrapper import UIWrapper
from modules.processing import StableDiffusionProcessing
from modules import prompt_parser, extra_networks, sd_hijack

class EmbedsScript(UIWrapper):
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
        pass

    def process_batch(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        active = getattr(p, 'embeds_active', active)
        if not active:
            return

        concept_list = prompt_utils.parse_concept_prompt(p.prompt)
        #concepts = [prompt_parser.parse_prompt_attention(concept)[0] for concept in concept_list]
        #concepts = [[concept, 1.0] for concept in concepts]


        concept_conds = []
        for concept in concept_list:
            text, _ = extra_networks.parse_prompt(p.prompt)

            _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])

            prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, p.steps)

            flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
            prompts = [prompt_text for step, prompt_text in flat_prompts]
            token_count, max_length = max([sd_hijack.model_hijack.get_prompt_lengths(prompt) for prompt in prompts], key=lambda args: args[0])

            #for concept, strength in concepts:
                #prompt_list = [concept] * p.batch_size
                #prompt_list = [concept] * 2
            prompts = prompt_parser.SdConditioning(concept_list, width=p.width, height=p.height)
            c = p.get_conds_with_caching(prompt_parser.get_multicond_learned_conditioning, prompts, p.steps, [self.cached_c], p.extra_network_data)
            concept_conds += list(zip(concept_list, c.batch))


        
        pass


    def postprocess_batch(self, p, *args, **kwargs):
        pass
    
    def unhook_callbacks(self) -> None:
        pass

    def get_xyz_axis_options(self) -> dict:
        return {}
