import gradio as gr
import logging
import torch
import torchvision.transforms as F
from modules import shared, scripts, devices, patches, script_callbacks
from modules.devices import NansException
from modules.script_callbacks import CFGDenoiserParams
from modules.processing import StableDiffusionProcessing
from scripts.incantation_base import UIWrapper
from scripts.scfg import scfg_combine_denoised
from scripts.adaptive_projected_guidance import normalized_guidance

logger = logging.getLogger(__name__)


class CFGCombinerParams:
        def __init__(self):
                self.current_step = 0


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
                "scfg_params": None,
                "cfgi_params": None,
                "tcg_params": None,
                "apg_params": None,
                "sg_params": None
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
            scfg_active = p.extra_generation_params.get('SCFG Active', False)
            cfgi_active = p.extra_generation_params.get('CFG Interval Enable', False) or p.extra_generation_params.get('EP-CFG Enable', False) or p.extra_generation_params.get('TCFG Enable', False)
            tcg_active = p.extra_generation_params.get('TCG Active', False)
            apg_active = p.extra_generation_params.get('APG Active', False)
            sg_active = p.extra_generation_params.get('SG Active', False)

            if not any([
                        pag_active,
                        scfg_active,
                        cfgi_active,
                        sg_active,
                        tcg_active,
                        apg_active
                    ]):
                return

            #logger.debug("CFGCombinerScript process_batch: pag_active or scfg_active")
            cfg_params = CFGCombinerParams()
            cfg_denoise_lambda = lambda params: self.on_cfg_denoiser_callback(params, p.incant_cfg_params, cfg_params)
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

        def on_cfg_denoiser_callback(self, params: CFGDenoiserParams, cfg_dict: dict, cfg_params: CFGCombinerParams):
            """ Callback for when the CFG denoiser is called 
            Patches the combine_denoised function with a custom one.
            """
            cfg_params.current_step = params.sampling_step

            if cfg_dict['denoiser'] is None:
                    cfg_dict['denoiser'] = params.denoiser
            else:
                    self.unpatch_cfg_denoiser(cfg_dict)
            self.patch_cfg_denoiser(params.denoiser, cfg_dict, cfg_params)

        def patch_cfg_denoiser(self, denoiser, cfg_dict: dict, cfg_params: CFGCombinerParams):
            """ Patch the CFG Denoiser combine_denoised function """
            if not cfg_dict:
                    logger.error("Unable to patch CFG Denoiser, no dict passed as cfg_dict")
                    return
            if not denoiser:
                    logger.error("Unable to patch CFG Denoiser, denoiser is None")
                    return
            if not cfg_params:
                    logger.error("Unable to patch CFG Denoiser, cfg_params is None")
                    return
            if getattr(denoiser, 'combine_denoised_patched', False) is False:
                    try:
                            setattr(denoiser, 'combine_denoised_original', denoiser.combine_denoised)
                            # create patch that references the original function
                            pass_conds_func = lambda *args, **kwargs: combine_denoised_pass_conds_list(
                                    *args,
                                    **kwargs,
                                    cfg_params = cfg_params,
                                    original_func = denoiser.combine_denoised_original,
                                    pag_params = cfg_dict['pag_params'],
                                    scfg_params = cfg_dict['scfg_params'],
                                    cfgi_params = cfg_dict['cfgi_params'],
                                    tcg_params = cfg_dict['tcg_params'],
                                    apg_params = cfg_dict['apg_params'],
                                    sg_params = cfg_dict['sg_params']
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
            1. CFG and S-CFG are combined multiplicatively
            2. PAG guidance is added to the result
            3. ...
            ...
        """
        cfg_params = kwargs.get('cfg_params', None)
        original_func = kwargs.get('original_func', None)
        pag_params = kwargs.get('pag_params', None)
        scfg_params = kwargs.get('scfg_params', None)
        cfgi_params = kwargs.get('cfgi_params', None)
        apg_params = kwargs.get('apg_params', None)
        tcg_params = kwargs.get('tcg_params', None)
        sg_params = kwargs.get('sg_params', None)

        if not any([
                pag_params,
                scfg_params,
                cfgi_params,
                apg_params,
                tcg_params,
                sg_params
        ]):
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
                if cfgi_params is not None:
                        if cfgi_params.cfg_interval_enable:
                                cfg_scale = cfgi_params.cfg_interval_scheduled_value

                # 2. PAG
                pag_x_out = None
                pag_scale = None
                run_pag = False
                if pag_params is not None:
                        pag_active = pag_params.pag_active
                        pag_x_out = pag_params.pag_x_out
                        pag_scale = pag_params.pag_scale

                        if not pag_active:
                                pass
                        # Not within step interval? 
                        elif not pag_params.pag_start_step <= pag_params.step <= pag_params.pag_end_step:
                                pass
                        # Scale is zero?
                        elif pag_scale <= 0:
                                pass
                        else:
                                run_pag = pag_active

                # 3. Saliency Map
                use_saliency_map = False
                if pag_params is not None:
                        use_saliency_map = pag_params.pag_sanf

                ### Combine Denoised
                for i, conds in enumerate(conds_list):
                        for cond_index, weight in conds:
                                if cfgi_params is not None:
                                        # 7. TCFG [arXiv:2503.18137] Kwon et. al.
                                        if cfgi_params.tcfg_enable:
                                                uncond = denoised[i].reshape(-1) # (D,)
                                                cond = x_out[cond_index].reshape(-1) # (D,)

                                                A = torch.stack([uncond, cond], dim=0) # (2, D)
                                                U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                                                # U: (2, 2) 
                                                # S: (2,)
                                                # Vh: (2, D)
                                                v1 = Vh[0, :] # (D,)
                                                dotval = torch.dot(uncond,v1)
                                                uncond_proj = dotval * v1
                                                # reshape to (C, H, W)
                                                uncond_proj = uncond_proj.view_as(denoised[i])
                                                denoised[i] = uncond_proj

                                model_delta = x_out[cond_index] - denoised[i]
                                #model_delta = x_out[cond_index] - denoised_uncond[i]

                                # 1. Experimental formulation for S-CFG combined with CFG combined with APG
                                cfg_o = model_delta * (weight * cfg_scale) # original delta
                                cfg_x = cfg_o.detach().clone() # modified

                                # 1. PAG
                                # PAG is added like CFG
                                if pag_params is not None:
                                        if not run_pag:
                                                pass
                                        # do pag
                                        else:
                                                try:
                                                        pag_delta = x_out[cond_index] - pag_x_out[i]
                                                        pag_x = pag_delta * (weight * pag_scale)

                                                        if pag_params.pag_sanf:
                                                                sal_cfg = sanf(cfg_o, pag_x)
                                                                cfg_x += sal_cfg
                                                        else:
                                                                cfg_x += pag_x

                                                except Exception as e:
                                                        logger.exception("Exception in combine_denoised_pass_conds_list - %s", e)

                                # 2. TCG
                                # TCG is added like CFG
                                if tcg_params is not None:
                                        if not tcg_params.tcg_active or tcg_params.tcg_scale <= 0 or tcg_params.tcg_x_out is None \
                                                or not tcg_params.tcg_start_step <= tcg_params.step <= tcg_params.tcg_end_step:
                                                pass
                                        else:
                                                try:
                                                        tcg_delta = x_out[cond_index] - tcg_params.tcg_x_out[i]
                                                        tcg_x = tcg_delta * (weight * tcg_params.tcg_scale)

                                                        if tcg_params.tcg_sanf:
                                                                sal_tcg = sanf(cfg_o, tcg_x)
                                                                cfg_x += sal_tcg
                                                        else:
                                                                cfg_x += tcg_x

                                                except Exception as e:
                                                        logger.exception("Exception in combine_denoised_pass_conds_list - %s", e)

                                # 3. Self-Guidance
                                if sg_params is not None:
                                        if not sg_params.sg_active or not sg_params.sg_start_step <= sg_params.step <= sg_params.sg_end_step or sg_params.sg_scale == 0 or sg_params.sg_x_out is None:
                                                pass
                                        # do pag
                                        else:
                                                try:
                                                        sg_delta = x_out[cond_index] - sg_params.sg_x_out[i]
                                                        sg_x = sg_delta * (weight * sg_params.sg_scale)

                                                        if sg_params.sg_sanf:
                                                                sal_sg = sanf(cfg_o, sg_x)
                                                                cfg_x += sal_sg
                                                        else:
                                                                cfg_x += sg_x

                                                except Exception as e:
                                                        logger.exception("Exception in combine_denoised_pass_conds_list - %s", e)

                                # 4. S-CFG
                                rate = 1.0
                                if scfg_params is not None:
                                        rate = scfg_combine_denoised(
                                                        model_delta = cfg_x,
                                                        cfg_scale = cfg_scale,
                                                        scfg_params = scfg_params,
                                        )
                                        # If rate is not an int, convert to tensor
                                        if rate is None:
                                               logger.error("scfg_combine_denoised returned None, using default rate of 1.0")
                                               rate = 1.0
                                        elif not isinstance(rate, int) and not isinstance(rate, float):
                                               rate = rate.to(device=shared.device, dtype=model_delta.dtype)
                                        else:
                                               # rate is tensor, probably
                                               pass
                                cfg_x = rate * cfg_x

                                # 5. APG
                                if apg_params is not None:
                                        if apg_params.apg_start_step <= cfg_params.current_step <= apg_params.apg_end_step:
                                                cfg_x = (cfg_scale-1) * normalized_guidance(
                                                        pred_cond=cfg_x,
                                                        pred_uncond=denoised_uncond[i],
                                                        apg_params = apg_params,
                                                        index = i,
                                                )

                                # 6. EP-CFG
                                # Isolate the latents between 0.45 and 0.55 in the energy histogram
                                # Rescale the cfg term by sqrt of the energy of the original prediction by the energy of the denoised prediction
                                if cfgi_params is not None:
                                        if cfgi_params.ep_cfg_enable:
                                                min_p = cfgi_params.ep_cfg_min
                                                max_p = cfgi_params.ep_cfg_max
                                                xc = x_out[cond_index]
                                                #xc = x_out[cond_index]
                                                xcfg = x_out[cond_index] + cfg_x
                                                ...
                                                # Step 2: Calculate robust energy for xc

                                                #xc_energy = torch.norm(xc, dim=(1, 2))**2
                                                b, h, w = xc.shape
                                                xc_energy = torch.norm(xc, dim=(1, 2))**2
                                                xc_energy = torch.reshape(xc_energy, (xc.shape[0], -1))
                                                xc_energy = xc_energy.float()
                                                #xc_energy = xc_energy ** 2
                                                q_45_xc = torch.quantile(xc_energy, min_p, dim=-1, keepdim=True)
                                                q_55_xc = torch.quantile(xc_energy, max_p, dim=-1, keepdim=True)
                                                mask_xc = (xc_energy >= q_45_xc) & (xc_energy <= q_55_xc)
                                                robust_energy_xc = torch.sum(xc_energy * mask_xc.float(), dim=-1)
                                                
                                                # Step 3: Calculate robust energy for xcfg
                                                xcfg_energy = torch.norm(xcfg, dim=(1, 2))**2
                                                xcfg_energy = torch.reshape(xcfg_energy, (xcfg.shape[0], -1))
                                                xcfg_energy = xcfg_energy.float()
                                                #xcfg_energy = xcfg_energy ** 2
                                                q_45_xcfg = torch.quantile(xcfg_energy, min_p, dim=-1, keepdim=True)
                                                q_55_xcfg = torch.quantile(xcfg_energy, max_p, dim=-1, keepdim=True)
                                                mask_xcfg = (xcfg_energy >= q_45_xcfg) & (xcfg_energy <= q_55_xcfg)
                                                robust_energy_xcfg = torch.sum(xcfg_energy * mask_xcfg.float(), dim=-1)
                                                
                                                # Step 4: Rescale xcfg based on the ratio of robust energies
                                                scaling_factor = torch.sqrt(robust_energy_xc / (robust_energy_xcfg + 1e-6))
                                                cfg_x = cfg_x * scaling_factor.unsqueeze(-1).unsqueeze(-1)

                                # 6. Add to denoised
                                denoised[i] += cfg_x
                                

                                devices.torch_gc()

                return denoised

        return new_combine_denoised(*args)

# 3. Saliency Adaptive Noise Fusion arXiv.2311.10329v5
def sanf(cfg_x, pag_x):
        blur = F.GaussianBlur(kernel_size=3, sigma=1).to(device=shared.device)
        omega_rt = blur(torch.abs(cfg_x))
        omega_rs = blur(torch.abs(pag_x))
        soft_rt = torch.softmax(omega_rt, dim=0)
        soft_rs = torch.softmax(omega_rs, dim=0)

        m = torch.stack([soft_rt, soft_rs], dim=0) # 2 c h w
        _, argmax_indices = torch.max(m, dim=0)

        # select from cfg_x or pag_x
        m1 = torch.where(argmax_indices == 0, 1, 0)

        # hadamard product
        sal_cfg = cfg_x * m1 + pag_x * (1 - m1)
        return sal_cfg