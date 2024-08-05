import os
import logging
import copy
import gradio as gr
import torch
import re
from torchvision.transforms import GaussianBlur


from einops import rearrange
from modules import shared, script_callbacks
from modules.images import get_next_sequence_number
from modules.processing import StableDiffusionProcessing
from scripts.ui_wrapper import UIWrapper, arg
from scripts.incant_utils import module_hooks, plot_tools, prompt_utils

logger = logging.getLogger(__name__)


module_field_map = {
    'savemaps': True,
    'savemaps_batch': None,
    'savemaps_step': None,
    'savemaps_save_steps': None,
}


SUBMODULES = ['to_q', 'to_k', 'to_v']


class SaveAttentionMapsScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []

    def title(self) -> str:
        return "Save Attention Maps"
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion('Save Attention Maps', open = False):
            with gr.Row():
                active = gr.Checkbox(label = 'Active', default = False)
                map_types = gr.CheckboxGroup(
                    label = 'Map Types',
                    choices = ['One-Hot Map', 'Per-Token Maps'],
                    value = ['One-Hot Map'],
                    info = 'Select the type of attention maps to save.',
                )
            export_folder = gr.Textbox(visible=False, label = 'Export Folder', value = 'attention_maps', info = 'Folder to save attention maps to as a subdirectory of the outputs.')
            module_name_filter = gr.Textbox(label = 'Module Names', value = 'input_blocks_5_1_transformer_blocks_0_attn2', info = 'Module name to save attention maps for. If the substring is found in the module name, the attention maps will be saved for that module.')
            class_name_filter = gr.Textbox(label = 'Class Name Filter', value = 'CrossAttention', info = 'Filters eligible modules by the class name.')
            save_every_n_step = gr.Slider(label = 'Save Every N Step', value = 0, min = 0, max = 100, step = 1, info = 'Save attention maps every N steps. 0 to save last step.')
            print_modules = gr.Button(value = 'Print Modules To Console')
            print_modules.click(self.print_modules, inputs=[module_name_filter, class_name_filter])

        self.infotext_fields = []
        self.paste_field_names = []

        opts = [active, module_name_filter, class_name_filter, save_every_n_step, map_types]
        for opt in opts:
            opt.do_not_save_to_config = True
        return opts
    
    def before_process_batch(self, p: StableDiffusionProcessing, active, module_name_filter, class_name_filter, save_every_n_step, map_types, *args, **kwargs):
        # Always unhook the modules first
        module_list = self.get_modules_by_filter(module_name_filter, class_name_filter)
        script_callbacks.remove_current_script_callbacks()
        self.unhook_modules(module_list, copy.deepcopy(module_field_map))

        setattr(p, 'savemaps_module_list', module_list)
        setattr(p, 'savemaps_map_types', map_types)

        if not active:
            return
        
        token_count, _= prompt_utils.get_token_count(p.prompt, p.steps, True)

        if token_count <= 0:
            logger.warning("No tokens found in prompt. Skipping saving attention maps.")
            return

        setattr(p, 'savemaps_token_count', token_count)
        setattr(p, 'savemaps_step', 0)

        token_indices = []
        # Tokenize/decode the prompts
        tokenized_prompts = []
        batch_chunks, _ = prompt_utils.tokenize_prompt(p.prompt)
        for batch in batch_chunks:
            for sub_batch in batch:
                tokenized_prompts.append(prompt_utils.decode_tokenized_prompt(sub_batch.tokens))
        for tp_prompt in tokenized_prompts:
            for tp in tp_prompt:
                token_idx, token_id, word = tp
                # jank
                if token_id < 49406:
                    token_indices.append(token_idx)
                # sanitize tokenized prompts
                tp[2] = re.escape(word)


        setattr(p, 'savemaps_tokenized_prompts', tokenized_prompts)
        setattr(p, 'savemaps_token_indices', token_indices)

                
        # Make sure the output folder exists
        outpath_samples = p.outpath_samples
        # Move this to plot tools?
        if not outpath_samples:
            logger.warning("No output path found. Skipping saving attention maps.")
            return
        output_folder_path = os.path.join(outpath_samples, 'attention_maps')
        if not os.path.exists(output_folder_path):
            logger.info(f"Creating directory: {output_folder_path}")
            os.makedirs(output_folder_path)
        
        # sequence number for saving
        seq_num = get_next_sequence_number(output_folder_path, basename='')
        setattr(p, 'savemaps_seq_num', seq_num)

        latent_shape = [p.height // p.rng.shape[1], p.width // p.rng.shape[2]] # (height, width)
        
        save_steps = []
        min_step = max(save_every_n_step-1, 0) 
        if save_every_n_step > 0:
            save_steps = list(range(min_step, p.steps, save_every_n_step))
        else:
            save_steps = [p.steps-1]
        # always save last step
        if p.steps-1 not in save_steps:
            save_steps.append(p.steps-1)
        setattr(p, 'savemaps_save_steps', save_steps)

        # Create fields in module
        value_map = copy.deepcopy(module_field_map)
        value_map['savemaps_save_steps'] = save_steps
        value_map['savemaps_step'] = 0
        #value_map['savemaps_shape'] = torch.tensor(latent_shape).to(device=shared.device, dtype=torch.int32)
        self.hook_modules(module_list, value_map, p)
        self.create_save_hook(module_list)

        def on_cfg_denoiser(params: script_callbacks.CFGDenoiserParams):
            """ Sets the step for all modules
                the webui reports an incorrect step so we just count it ourselves
            """
            for module in module_list:
                module.savemaps_step = p.savemaps_step
            # logger.debug('Setting step to %d for %d modules', p.savemaps_step, len(module_list))
            p.savemaps_step += 1
        
        script_callbacks.on_cfg_denoiser(on_cfg_denoiser)


    def process(self, p, *args, **kwargs):
        pass

    def before_process(self, p: StableDiffusionProcessing, active, module_name_filter, class_name_filter, save_every_n_step, map_types, *args, **kwargs):
        module_list = self.get_modules_by_filter(module_name_filter, class_name_filter)
        self.unhook_modules(module_list, copy.deepcopy(module_field_map))

    def process_batch(self, p, *args, **kwargs):
        pass

    def postprocess_batch(self, p: StableDiffusionProcessing, active, module_name_filter, class_name_filter, save_every_n_step, map_types, *args, **kwargs):
        module_list = self.get_modules_by_filter(module_name_filter, class_name_filter)

        if getattr(p, 'savemaps_token_count', None) is None:
            self.unhook_modules(module_list, copy.deepcopy(module_field_map))
            return

        base_seq_num = getattr(p, 'savemaps_seq_num', None)
        map_types = getattr(p, 'savemaps_map_types', [])
        tokenized_prompts = getattr(p, 'savemaps_tokenized_prompts', None)
        token_indices = getattr(p, 'savemaps_token_indices', None)
        save_steps = getattr(p, 'savemaps_save_steps', None)
        save_image_path = os.path.join(p.outpath_samples, 'attention_maps')

        plot_is_self = False # kind of useless

        for module in module_list:
            network_layer_name = module.network_layer_name

            if not hasattr(module, 'savemaps_batch') or module.savemaps_batch is None:
                logger.error(f"No attention maps found for module: {network_layer_name}")
                continue

            # self attn maps are kind of useless atm
            is_self = getattr(module, 'savemaps_is_self', False)
            if is_self and not plot_is_self:
                continue

            # selfattn: seq_len = hw
            # crossattn: seq_len = # of tokens
            attn_maps = module.savemaps_batch # (attn_map num, 2 * batch_num, height * width, sequence_len)
            attn_map_num, batch_num, hw, seq_len = attn_maps.shape
            token_indices = p.savemaps_token_indices
            save_steps = p.savemaps_save_steps
            downscale_h = round((hw * (p.height / p.width)) ** 0.5)
            downscale_w = hw // downscale_h
            gaussian_blur = GaussianBlur(kernel_size=3, sigma=1)

            # Blur maps
            if is_self:
                attn_maps = attn_maps.view(attn_map_num * batch_num, downscale_h, downscale_w, seq_len) # if self-attn, we need to blur over the sequence length
            attn_maps = attn_maps.permute(0, 3, 1, 2) # (ab, seq_len, height, width)
            attn_maps = gaussian_blur(attn_maps)  # Applying Gaussian smoothing
            attn_maps = attn_maps.permute(0, 2, 3, 1) # (ab, height, width, seq_len)
            if is_self:
                attn_maps = attn_maps.view(attn_map_num, 2, batch_num // 2, downscale_h * downscale_w, seq_len).mean(dim=1) # (attn_map num, batch_num, hw, hw)
                attn_maps = attn_maps.unsqueeze(2) # (attn_map num, batch_num, 1, hw, hw)
            else:
                attn_maps = rearrange(attn_maps, 'n (m b) (h w) t -> n m b t h w', m = 2, h = downscale_h).mean(dim=1) # (attn_map num, batch_num, token_idx, height, width)
                attn_map_num, batch_num, token_dim, h, w = attn_maps.shape

            output_dict_maps = []
            per_token_dict_maps = []
            one_hot_dict_maps = []

            if 'Per-Token Maps' in map_types:

                # write to dict
                for attn_map_idx in range(attn_maps.shape[0]):
                    for batch_idx in range(batch_num):
                        for token_idx in token_indices:

                            attnmap = attn_maps[attn_map_idx, batch_idx, token_idx]
                            _, token_id, word = tokenized_prompts[batch_idx][token_idx]

                            plot_type = f"({token_idx}, {token_id}, '{word}')"
                            filename_info = f'token{token_idx:04}'
                            plot_color = 'viridis'

                            map_info: dict = self.create_base_dict(plot_type, base_seq_num, network_layer_name, save_steps, attn_map_idx, batch_idx, attnmap, filename_info, plot_color)
                            map_info.update({
                                'token_idx': token_idx,
                                'token_id': token_id,
                                'token_word': word,
                            })
                            output_dict_maps.append(map_info)

            if 'One-Hot Map' in map_types:
                one_hot_map = attn_maps[:, :, token_indices] # (attn_map num, batch_num, token_idx, height, width)
                one_hot_map = one_hot_map.argmax(dim=2, keepdim=True)
                one_hot_map = one_hot_map.to(dtype=torch.float16)

                # quantize to stable number of colors s.t. 
                num_colors = max(len(token_indices), 1)
                min_val, max_val = one_hot_map.min(), one_hot_map.max()
                step = 1 / num_colors
                one_hot_map *= step
                one_hot_map = one_hot_map.sum(dim=2) # (attn_map num, batch_num, height, width)

                # write to dict
                for attn_map_idx in range(one_hot_map.shape[0]):
                    for batch_idx in range(batch_num):
                        plot_type = "One Hot"
                        plot_color = 'plasma'
                        attnmap = one_hot_map[attn_map_idx, batch_idx]
                        ohm_info: dict = self.create_base_dict(plot_type, base_seq_num, network_layer_name, save_steps, attn_map_idx, batch_idx, attnmap, 'ohm', plot_color)
                        output_dict_maps.append(ohm_info)

            # Save maps from map dict
            for md in output_dict_maps:
                base_seq_num = md['seq_num']
                network_layer_name = md['network_layer_name']
                savestep_num = md['savestep_num']
                attn_map_idx = md['attn_map_idx']
                batch_idx = md['batch_idx']

                # output filename and path
                filename_info = md['filename_info']
                if len(filename_info) > 0:
                    filename_info = f'{filename_info}_'

                out_file_name = f'{base_seq_num:04}-{network_layer_name}_{filename_info}step{savestep_num:04}_attnmap_{attn_map_idx:04}_batch{batch_idx:04}.png'
                out_save_path = os.path.join(save_image_path, out_file_name)

                # plot title
                plot_type = md['plot_type']
                plot_color = md['plot_color']
                plot_title = f"{network_layer_name}\nStep {savestep_num}"
                if len(plot_type) > 0:
                    plot_title += f", {plot_type}"

                attn_map = md['attnmap']
                plot_tools.plot_attention_map(
                    attention_map = attn_map,
                    title = plot_title,
                    save_path = out_save_path,
                    plot_type = plot_color,
                )

                if shared.state.interrupted:
                    self.unhook_modules(module_list, copy.deepcopy(module_field_map))
                    return 
        self.unhook_modules(module_list, copy.deepcopy(module_field_map))

    def create_base_dict(self, plot_type:str, base_seq_num: int, network_layer_name: str, save_steps: list, attn_map_idx: int, batch_idx: int, attnmap: torch.Tensor, filename_info: str, plot_color: str):
        """ Create a base dictionary for saving attention maps for minimum metadata that the save function expects 
        Arguments:
                plot_type: str - name of the type of plot, used in the plot title
                base_seq_num: int - start sequence number for saving, prefixes the filename with "000xx-" where xx is the sequence number
                module_name: str - the module's network layer name
                save_steps: list[int] - list of steps to save attention maps for, should be same length as the number of attention maps
                attn_map_idx: int - index of the attention map
                batch_idx: int - index of the batch
                attnmap: torch.Tensor - attention map of shape [C, H, W]
                filename_info: str- a string that goes in the middle of the filename f"000xx-{filename_info}-000yy.png"
                plot_color: str - one of the matplotlib color maps (default is 'viridis')
        """
        network_layer_name = network_layer_name.removeprefix('diffusion_model_')
        network_layer_name = network_layer_name.replace('transformer_blocks_', 'tr_bl_')
        base_dict = {
            'plot_type': plot_type,
            'seq_num': base_seq_num + batch_idx,
            'step': save_steps[attn_map_idx] + 1,
            'network_layer_name': network_layer_name,
            'attn_map_idx': attn_map_idx,
            'savestep_num': save_steps[attn_map_idx] + 1,
            'batch_idx': batch_idx,
            'attnmap': attnmap,
            'filename_info': filename_info,
            'plot_color': plot_color,
        }
        return base_dict
    
    def unhook_callbacks(self) -> None:
        pass

    def get_xyz_axis_options(self) -> dict:
        return {}
    
    def get_infotext_fields(self) -> list:
        return self.infotext_fields
    
    def create_save_hook(self, module_list):
        pass

    def hook_modules(self, module_list: list, value_map: dict, p: StableDiffusionProcessing):
        def savemaps_hook(module, input, kwargs, output):
            """ Hook to save attention maps every N steps, or the last step if N is 0.
            Saves attention maps to a field named 'savemaps_batch' in the module.
            with shape (attn_map, batch_num, height * width).
            
            """
            #module.savemaps_step += 1

            if not module.savemaps_step in module.savemaps_save_steps:
                return
            reweight_crossattn = True 


            is_self = getattr(module, 'savemaps_is_self', False)
            to_q_map = getattr(module, 'savemaps_to_q_map', None)
            to_k_map = to_q_map if module.savemaps_is_self else getattr(module, 'savemaps_to_k_map', None)

            # we want to reweight the attention scores by removing influence of the first token
            orig_seq_len = to_k_map.shape[1]
            # token_count = module.savemaps_token_count
            # min_token = 0
            # max_token = min(token_count+1, orig_seq_len)
            token_indices = module.savemaps_token_indices

            if not is_self and reweight_crossattn:
                to_k_map = to_k_map[:, token_indices, :]

            attn_map = get_attention_scores(to_q_map, to_k_map, dtype=to_q_map.dtype)
            b, hw, seq_len = attn_map.shape

            if not is_self and reweight_crossattn:
                #to_attn_zeros = torch.zeros([b, hw]).unsqueeze(-1).to(device=shared.device, dtype=attn_map.dtype) # (batch, h*w, 1)
                #attn_map = torch.cat([to_attn_zeros, attn_map], dim=-1) # re pad to original token dim size
                left_pad = 1
                right_pad = orig_seq_len - seq_len - 1
                attn_map = torch.nn.functional.pad(attn_map, (left_pad, right_pad), value=0) # re pad to original token dim size

            # multiply into text embeddings
            attn_map = attn_map.unsqueeze(0)

            #attn_map = attn_map.mean(dim=-1)
            if module.savemaps_batch is None:
                module.savemaps_batch = attn_map
            else:
                module.savemaps_batch = torch.cat([module.savemaps_batch, attn_map], dim=0)

        def savemaps_to_q_hook(module, input, kwargs, output):
                setattr(module.savemaps_parent_module[0], 'savemaps_to_q_map', output)

        def savemaps_to_k_hook(module, input, kwargs, output):
                if not module.savemaps_parent_module[0].savemaps_is_self:
                    setattr(module.savemaps_parent_module[0],'savemaps_to_k_map', output)

        def savemaps_to_v_hook(module, input, kwargs, output):
                setattr(module.savemaps_parent_module[0],'savemaps_to_v_map', output)

        #for module, kv in zip(module_list, value_map.items()):
        for module in module_list:
            # logger.debug('Adding hook to %s', module.network_layer_name)
            for key_name, default_value in value_map.items():
                module_hooks.modules_add_field(module, key_name, default_value)

            module_hooks.module_add_forward_hook(module, savemaps_hook, 'forward', with_kwargs=True)
            module_hooks.modules_add_field(module, 'savemaps_token_count', p.savemaps_token_count)
            module_hooks.modules_add_field(module, 'savemaps_token_indices', p.savemaps_token_indices)

            if module.network_layer_name.endswith('attn1'): # self attn
                module_hooks.modules_add_field(module, 'savemaps_is_self', True)
            if module.network_layer_name.endswith('attn2'): # self attn
                module_hooks.modules_add_field(module, 'savemaps_is_self', False)

            for module_name in SUBMODULES:
                if not hasattr(module, module_name):
                    logger.error(f"Submodule not found: {module_name} in module: {module.network_layer_name}")
                    continue
                submodule = getattr(module, module_name)
                hook_fn_name = f'savemaps_{module_name}_hook'
                hook_fn = locals().get(hook_fn_name, None)
                if not hook_fn:
                    logger.error(f"Hook function '{hook_fn_name}' not found for submodule: {module_name}")
                    continue

                module_hooks.modules_add_field(submodule, 'savemaps_parent_module', [module])
                module_hooks.module_add_forward_hook(submodule, hook_fn, 'forward', with_kwargs=True)
    
    def unhook_modules(self, module_list: list, value_map: dict):
        for module in module_list:
            for key_name, _ in value_map.items():
                module_hooks.modules_remove_field(module, key_name)
            module_hooks.modules_remove_field(module, 'savemaps_is_self')
            module_hooks.modules_remove_field(module, 'savemaps_token_count')
            module_hooks.modules_remove_field(module, 'savemaps_token_indices')
            module_hooks.remove_module_forward_hook(module, 'savemaps_hook')
            for module_name in SUBMODULES:
                module_hooks.modules_remove_field(module, f'savemaps_{module_name}_map')

                if hasattr(module, module_name):
                    submodule = getattr(module, module_name)
                    module_hooks.modules_remove_field(submodule, 'savemaps_parent_module')    
                    module_hooks.remove_module_forward_hook(submodule, f'savemaps_{module_name}_hook')


    def print_modules(self, module_name_filter, class_name_filter):
            logger.info("Module name filter: '%s', Class name filter: '%s'", module_name_filter, class_name_filter)
            modules = self.get_modules_by_filter(module_name_filter, class_name_filter)
            module_names = [""]
            if len(modules) > 0:
                module_names = "\n".join([f"{m.network_layer_name}: {m.__class__.__name__}" for m in modules])
            logger.info("Modules found:\n----------\n%s\n----------\n", module_names)

    def get_modules_by_filter(self, module_name_filter, class_name_filter):
        if len(class_name_filter) == 0:
            class_name_filter = None
        if len(module_name_filter) == 0:
            module_name_filter = None
        found_modules = module_hooks.get_modules(module_name_filter, class_name_filter)
        if len(found_modules) == 0:
            logger.warning(f"No modules found with module name filter: {module_name_filter} and class name filter")
        return found_modules


def get_attention_scores(to_q_map, to_k_map, dtype):
        """ Calculate the attention scores for the given query and key maps
        Arguments:
                to_q_map: torch.Tensor - query map
                to_k_map: torch.Tensor - key map
                dtype: torch.dtype - data type of the tensor
        Returns:
                torch.Tensor - attention scores 
        """
        # based on diffusers models/attention.py "get_attention_scores"
        # use in place operations vs. softmax to save memory: https://stackoverflow.com/questions/53732209/torch-in-place-operations-to-save-memory-softmax
        # 512x: 2.65G -> 2.47G

        attn_probs = to_q_map @ to_k_map.transpose(-1, -2)
        attn_probs = attn_probs.to(dtype=torch.float32) #

        channel_dim = to_q_map.size(1)
        attn_probs /= (channel_dim ** 0.5)
        attn_probs -= attn_probs.max()

        # avoid nan by converting to float32 and subtracting max 
        attn_probs = attn_probs.softmax(dim=-1).to(device=shared.device, dtype=to_q_map.dtype)
        attn_probs = attn_probs.to(dtype=dtype)

        return attn_probs