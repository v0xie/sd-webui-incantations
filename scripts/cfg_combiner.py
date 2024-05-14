import gradio as gr
import logging
from modules import scripts
from modules.processing import StableDiffusionProcessing
from scripts.incantation_base import UIWrapper

logger = logging.getLogger(__name__)

class CFGCombinerScript(UIWrapper):
        def __init__(self):
                pass

        # Extension title in menu UI
        def title(self):
                return "CFG Combiner"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def ui(self, is_img2img):
            return []
        
        def before_process(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript before_process")
            pass

        def process(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript process")
            pass

        def before_process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript before_process_batch")
            pass
        
        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript process_batch")
            pass

        def postprocess_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript postprocess_batch")
            pass

        def unhook_callbacks(self):
            pass