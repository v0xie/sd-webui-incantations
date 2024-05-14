import gradio as gr
import logging
import torch
from modules import scripts, patches, script_callbacks
from modules.script_callbacks import CFGDenoiserParams
from modules.processing import StableDiffusionProcessing
from scripts.incantation_base import UIWrapper

logger = logging.getLogger(__name__)

class CFGCombinerScript(UIWrapper):
        """ Some scripts modify the CFGs in ways that are not compatible with each other.
            This script will patch the CFG denoiser function to apply CFG in an ordered way.
            This script adds a dict named 'incant_cfg_params' to the processing object.
            This dict contains the following:
                'denoiser': the denoiser object
                'pag_params': list of PAG parameters
                'scfg_params': the S-CFG parameters
                ...
        """
        def __init__(self):
                pass

        # Extension title in menu UI
        def title(self):
                return "CFG Combiner"

        # Decide to show menu in txt2img or img2img
        def show(self, is_img2img):
                return scripts.AlwaysVisible

        # Setup menu ui detail
        def setup_ui(self, is_img2img):
            self.infotext_fields = []
            self.paste_field_names = []
            return []
        
        def before_process(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript before_process")
            cfg_dict = {
                "denoiser": None,
                "pag_params": None,
                "scfg_params": None
            }
            setattr(p, 'incant_cfg_params', cfg_dict)

        def process(self, p: StableDiffusionProcessing, *args, **kwargs):
            pass

        def before_process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            pass
        
        def process_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            """ Process the batch and hook the CFG denoiser if PAG or S-CFG is active """
            logger.debug("CFGCombinerScript process_batch")
            pag_active = p.extra_generation_params.get('PAG Active', False)
            scfg_active = p.extra_generation_params.get('S-CFG Active', False)

            if not any([
                        pag_active,
                        scfg_active
                    ]):
                return

            logger.debug("CFGCombinerScript process_batch: pag_active or scfg_active")

            cfg_denoise_lambda = lambda params: self.on_cfg_denoiser_callback(params, p.incant_cfg_params)
            unhook_lambda = lambda: self.unhook_callbacks()

            script_callbacks.on_cfg_denoiser(cfg_denoise_lambda)
            script_callbacks.on_script_unloaded(unhook_lambda)
            logger.debug('Hooked callbacks')

        def postprocess_batch(self, p: StableDiffusionProcessing, *args, **kwargs):
            logger.debug("CFGCombinerScript postprocess_batch")
            script_callbacks.remove_current_script_callbacks()

        def unhook_callbacks(self, cfg_dict = None):
            if not cfg_dict:
                    return
            self.unpatch_cfg_denoiser(cfg_dict)

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, cfg_dict: dict):
            """ Callback for when the CFG denoiser is called 
            Patches the combine_denoised function with a custom one.
            """
            if cfg_dict['denoiser'] is None:
                    cfg_dict['denoiser'] = params.denoiser
            else:
                    self.unpatch_cfg_denoiser(cfg_dict)
            self.patch_cfg_denoiser(params.denoiser, cfg_dict)

        def patch_cfg_denoiser(self, denoiser, cfg_dict: dict):
            """ Patch the CFG Denoiser combine_denoised function """
            if not cfg_dict:
                    logger.error("Unable to patch CFG Denoiser, no dict passed as cfg_dict")
                    return
            if not denoiser:
                    logger.error("Unable to patch CFG Denoiser, denoiser is None")
                    return

            if getattr(denoiser, 'combine_denoised_patched', False) is False:
                    try:
                            setattr(denoiser, 'combine_denoised_original', denoiser.combine_denoised)
                            # create patch that references the original function
                            pass_conds_func = lambda *args, **kwargs: combine_denoised_pass_conds_list(
                                    *args,
                                    **kwargs,
                                    original_func = denoiser.combine_denoised_original,
                                    pag_params = cfg_dict['pag_params'],
                                    scfg_params = cfg_dict['scfg_params']
                                )
                            patched_combine_denoised = patches.patch(__name__, denoiser, "combine_denoised", pass_conds_func)
                            setattr(denoiser, 'combine_denoised_patched', True)
                            setattr(denoiser, 'combine_denoised_original', patches.original(__name__, denoiser, "combine_denoised"))
                    except KeyError:
                            logger.exception("KeyError patching combine_denoised")
                            pass
                    except RuntimeError:
                            logger.exception("RuntimeError patching combine_denoised")
                            pass

        def unpatch_cfg_denoiser(self, cfg_dict = None):
            """ Unpatch the CFG Denoiser combine_denoised function """
            if cfg_dict is None:
                    return
            denoiser = cfg_dict.get('denoiser', None)
            if denoiser is None:
                    return

            setattr(denoiser, 'combine_denoised_patched', False)
            try:
                    patches.undo(__name__, denoiser, "combine_denoised")
            except KeyError:
                    logger.exception("KeyError unhooking combine_denoised")
                    pass
            except RuntimeError:
                    logger.exception("RuntimeError unhooking combine_denoised")
                    pass

            cfg_dict['denoiser'] = None


def combine_denoised_pass_conds_list(*args, **kwargs):
        """ Hijacked function for combine_denoised in CFGDenoiser 
        Currently relies on the original function not having any kwargs
        If any of the params are not None, it will apply the corresponding guidance
        The order of guidance is:
            1. Regular CFG guidance
            2. PAG guidance
            3. S-CFG guidance - not implemented yet
            ...
        """
        original_func = kwargs.get('original_func', None)
        pag_params = kwargs.get('pag_params', None)
        scfg_params = kwargs.get('scfg_params', None)

        if pag_params is None and scfg_params is None:
                logger.warning("No reason to hijack combine_denoised")
                return original_func(*args)

        def new_combine_denoised(x_out, conds_list, uncond, cond_scale):
                denoised_uncond = x_out[-uncond.shape[0]:]
                denoised = torch.clone(denoised_uncond)

                ### Variables
                # 0. Standard CFG Value
                cfg_scale = cond_scale

                # 1. CFG Interval
                # Overrides cfg_scale if pag_params is not None
                if pag_params is not None:
                        if pag_params.cfg_interval_enable:
                                cfg_scale = pag_params.cfg_interval_scheduled_value

                # 2. PAG
                pag_x_out = None
                pag_scale = None
                if pag_params is not None:
                        pag_x_out = pag_params.pag_x_out
                        pag_scale = pag_params.pag_scale

                ### Combine Denoised
                for i, conds in enumerate(conds_list):
                        for cond_index, weight in conds:

                            # 0/1. Regular CFG guidance / Scheduled CFG
                            denoised[i] += (x_out[cond_index] - denoised_uncond[i]) * (weight * cfg_scale)

                            # 2. PAG
                            # PAG is added like CFG
                            if pag_params is not None:
                                    # Within step interval? 
                                    if pag_params.pag_start_step <= pag_params.step <= pag_params.pag_end_step:
                                            pass
                                    # Scale non-zero?
                                    elif pag_scale <= 0:
                                            pass
                                    else:
                                            try:
                                                    denoised[i] += (x_out[cond_index] - pag_x_out[i]) * (weight * pag_scale)
                                            except Exception as e:
                                                    logger.exception("Exception in combine_denoised_pass_conds_list - %s", e)
                            
                            # 3. S-CFG
                            # Not implemented yet
                            if scfg_params is not None:
                                    pass

                return denoised
        return new_combine_denoised(*args)