import os
import logging
import gradio as gr

from modules.processing import StableDiffusionProcessing
from scripts.ui_wrapper import UIWrapper, arg
from scripts.incant_utils import module_hooks

logger = logging.getLogger(__name__)


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
    
    def before_process(self, p, active, module_name_filter, class_name_filter, save_every_n_step, *args, **kwargs):
        if not active:
            return
        outpath_samples = p.outpath_samples
        # move this to plot tools?
        if not outpath_samples:
            logger.warning("No output path found. Skipping saving attention maps.")
            return
        output_folder_path = os.path.join(outpath_samples, 'attention_maps')
        if not os.path.exists(output_folder_path):
            logger.info(f"Creating directory: {output_folder_path}")
            os.makedirs(output_folder_path)
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
    
    def get_infotext_fields(self) -> list:
        return self.infotext_fields

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

