import os
import logging
import copy
import gradio as gr
import torch


from modules import shared
from modules.processing import StableDiffusionProcessing
from scripts.ui_wrapper import UIWrapper, arg
from scripts.incant_utils import module_hooks, plot_tools

logger = logging.getLogger(__name__)


module_field_map = {
    'savemaps': True,
    'savemaps_batch': None,
    'savemaps_step': None,
    'savemaps_save_steps': None,
}


class SaveAttentionMapsScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []

    def title(self) -> str:
        return "Save Attention Maps"
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion('Save Attention Maps', open = False):
            active = gr.Checkbox(label = 'Active', default = False)
            export_folder = gr.Textbox(label = 'Export Folder', value = 'attention_maps', info = 'Folder to save attention maps to as a subdirectory of the outputs.')
            module_name_filter = gr.Textbox(label = 'Module Names', value = 'middle_block_1_transformer_blocks_0_attn1', info = 'Module name to save attention maps for. If the substring is found in the module name, the attention maps will be saved for that module.')
            class_name_filter = gr.Textbox(label = 'Class Name Filter', value = 'CrossAttention', info = 'Filters eligible modules by the class name.')
            save_every_n_step = gr.Slider(label = 'Save Every N Step', value = 0, min = 0, max = 100, step = 1, info = 'Save attention maps every N steps. 0 to save last step.')
            print_modules = gr.Button(value = 'Print Modules To Console')
            print_modules.click(self.print_modules, inputs=[module_name_filter, class_name_filter])
        active.do_not_save_to_config = True
        export_folder.do_not_save_to_config = True
        module_name_filter.do_not_save_to_config = True
        class_name_filter.do_not_save_to_config = True
        save_every_n_step.do_not_save_to_config = True
        self.infotext_fields = []
        self.paste_field_names = []
        return [active, module_name_filter, class_name_filter, save_every_n_step]
    
    def before_process_batch(self, p: StableDiffusionProcessing, active, module_name_filter, class_name_filter, save_every_n_step, *args, **kwargs):
        # Always unhook the modules first
        module_list = self.get_modules_by_filter(module_name_filter, class_name_filter)
        self.unhook_modules(module_list, copy.deepcopy(module_field_map))

        if not active:
            return

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

        latent_shape = [p.height // p.rng.shape[1], p.width // p.rng.shape[2]] # (height, width)
        
        save_steps = []
        min_step = max(save_every_n_step, 0) 
        max_step = max(p.steps+1, 0)
        if save_every_n_step > 0:
            save_steps = list(range(min_step, max_step, save_every_n_step))
        else:
            save_steps = [p.steps]

        # Create fields in module
        value_map = copy.deepcopy(module_field_map)
        value_map['savemaps_save_steps'] = torch.tensor(save_steps).to(device=shared.device, dtype=torch.int32)
        value_map['savemaps_step'] = torch.tensor([0]).to(device=shared.device, dtype=torch.int32)
        #value_map['savemaps_shape'] = torch.tensor(latent_shape).to(device=shared.device, dtype=torch.int32)
        self.hook_modules(module_list, value_map)
        self.create_save_hook(module_list)

    def process(self, p, *args, **kwargs):
        pass

    def before_process(self, p: StableDiffusionProcessing, active, module_name_filter, class_name_filter, save_every_n_step, *args, **kwargs):
        module_list = self.get_modules_by_filter(module_name_filter, class_name_filter)
        self.unhook_modules(module_list, copy.deepcopy(module_field_map))

    def process_batch(self, p, *args, **kwargs):
        pass

    def postprocess_batch(self, p: StableDiffusionProcessing, active, module_name_filter, class_name_filter, save_every_n_step, *args, **kwargs):
        module_list = self.get_modules_by_filter(module_name_filter, class_name_filter)

        save_image_path = os.path.join(p.outpath_samples, 'attention_maps')

        latent_shape = [p.height // p.rng.shape[1], p.width // p.rng.shape[2]] # (height, width)
        max_dims = p.height * p.width

        #max_dims = latent_shape[0] * latent_shape[1]
        for module in module_list:
            if not hasattr(module, 'savemaps_batch') or module.savemaps_batch is None:
                logger.error(f"No attention maps found for module: {module.network_layer_name}")
                continue
            attn_maps = module.savemaps_batch # (attn_map num, 2 * batch_num, height * width, sequence_len)

            attn_map_num, batch_num, hw, seq_len = attn_maps.shape

            downscale_ratio = max_dims / hw
            downscale_h = round((hw * (p.height / p.width)) ** 0.5)
            downscale_w = hw // downscale_h

            attn_maps = attn_maps.mean(dim=-1) # (attn_map num, batch_num, height * width)
            attn_maps = attn_maps.view(attn_map_num, 2, batch_num // 2, downscale_h, downscale_w).mean(dim=1)
            attn_map_num, batch_num, height, width = attn_maps.shape
            for i in range(attn_map_num):
                for j in range(batch_num):
                    savestep_num = module.savemaps_save_steps[i]
                    batch_idx = j
                    attn_map = attn_maps[i, j]
                    out_file_name = f'{module.network_layer_name}_attn_map_{i}_batch_{j}.png'
                    save_path = os.path.join(save_image_path, out_file_name)
                    plot_tools.plot_attention_map(
                        attention_map=attn_map,
                        title="",
                        save_path=save_path,
                        plot_type="default"
                    )

        self.unhook_modules(module_list, copy.deepcopy(module_field_map))
    
    def unhook_callbacks(self) -> None:
        pass

    def get_xyz_axis_options(self) -> dict:
        return {}
    
    def get_infotext_fields(self) -> list:
        return self.infotext_fields
    
    def create_save_hook(self, module_list):
        pass

    def hook_modules(self, module_list: list, value_map: dict):
        def savemaps_hook(module, input, kwargs, output):
            """ Hook to save attention maps every N steps, or the last step if N is 0.
            Saves attention maps to a field named 'savemaps_batch' in the module.
            with shape (attn_map, batch_num, height * width).
            
            """
            module.savemaps_step += 1

            #parent_module = getattr(module, 'savemaps_parent_module', None)
            #to_v_map = None
            #if parent_module is not None:
            to_v_map = getattr(module, 'savemaps_to_v_map', None)

            if (module.savemaps_step in module.savemaps_save_steps):
                #context = kwargs.get('context', None)
                attn_map = output.detach().clone()

                # multiply into text embeddings
                if to_v_map is not None:
                    attn_map = (to_v_map @ output.transpose(1,2)).transpose(1,2)
                
                attn_map = attn_map.unsqueeze(0)
                

                #attn_map = attn_map.mean(dim=-1)
                if module.savemaps_batch is None:
                    module.savemaps_batch = attn_map
                else:
                    module.savemaps_batch = torch.cat([module.savemaps_batch, attn_map], dim=0)

        def savemaps_to_v_hook(module, input, kwargs, output):
                module.savemaps_parent_module[0].savemaps_to_v_map = output

        #for module, kv in zip(module_list, value_map.items()):
        for module in module_list:
            for key_name, default_value in value_map.items():
                module_hooks.modules_add_field(module, key_name, default_value)
            module_hooks.module_add_forward_hook(module, savemaps_hook, 'forward', with_kwargs=True)
            if hasattr(module, 'to_v'):
                module_hooks.modules_add_field(module.to_v, 'savemaps_parent_module', [module])
                module_hooks.module_add_forward_hook(module.to_v, savemaps_to_v_hook, 'forward', with_kwargs=True)
    
    def unhook_modules(self, module_list: list, value_map: dict):
        for module in module_list:
            for key_name, _ in value_map.items():
                module_hooks.modules_remove_field(module, key_name)
            module_hooks.remove_module_forward_hook(module, 'savemaps_hook')
            if hasattr(module, 'to_v'):
                module_hooks.modules_remove_field(module.to_v, 'savemaps_parent_module')    
                module_hooks.remove_module_forward_hook(module.to_v, 'savemaps_to_v_hook')

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

