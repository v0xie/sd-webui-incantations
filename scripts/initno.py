import gradio as gr
import torch
from torch.nn import functional as F
import logging
import re
from scripts.ui_wrapper import UIWrapper
from modules import shared, script_callbacks
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
from modules.sd_samplers_cfg_denoiser import catenate_conds
from torch.distributions import Normal

from scripts.incant_utils import module_hooks, prompt_utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class InitnoParams():
    def __init__(self):
        self.x_in = None
        self.sigma_in = None
        self.image_cond_in = None
        self.text_cond = None
        self.text_uncond = None
        self.token_count = None
        self.max_length = None
        self.ran_once = False

class InitnoScript(UIWrapper):
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

    def before_process_batch(self, p, active, *args, **kwargs):
        self.unhook_callbacks()
        active = getattr(p, 'embeds_active', active)
        if not active:
            return
        
        def initno_to_q(module, input, kwargs, output):
            setattr(module.initno_parent_module[0], 'to_q_map', output)

        def initno_to_k(module, input, kwargs, output):
            setattr(module.initno_parent_module[0], 'to_k_map', output)

        def initno_cross_attn_hook(module, input, kwargs, output):
            q_map = module.to_q_map.detach().clone()
            k_map = module.to_k_map.detach().clone()
            attn_probs = get_attention_scores(q_map, k_map, dtype=q_map.dtype)
            setattr(module, 'initno_crossattn', attn_probs)

        def initno_self_attn_hook(module, input, kwargs, output):
            q_map = module.to_q_map.detach().clone()
            attn_probs = get_attention_scores(q_map, q_map, dtype=q_map.dtype)
            setattr(module, 'initno_selfattn', attn_probs)

        crossattn_modules = self.get_all_crossattn_modules()
        for module in crossattn_modules:
            module_hooks.modules_add_field(module.to_q, 'initno_parent_module', [module])
            module_hooks.modules_add_field(module.to_k, 'initno_parent_module', [module])
            #module_hooks.modules_add_field(module.to_out, 'initno_parent_module', [module])
            if module.network_layer_name.endswith('attn1'): # self attn
                module_hooks.module_add_forward_hook(module.to_q, initno_to_q, hook_type='forward', with_kwargs=True)
                module_hooks.module_add_forward_hook(module, initno_self_attn_hook, hook_type='forward', with_kwargs=True)
                module_hooks.modules_add_field(module, 'initno_selfattn', None)
            if module.network_layer_name.endswith('attn2'):
                module_hooks.module_add_forward_hook(module.to_q, initno_to_q, hook_type='forward', with_kwargs=True)
                module_hooks.module_add_forward_hook(module.to_k, initno_to_k, hook_type='forward', with_kwargs=True)
                module_hooks.module_add_forward_hook(module, initno_cross_attn_hook, hook_type='forward', with_kwargs=True)
                module_hooks.modules_add_field(module, 'initno_crossattn', None)
        
    def get_all_crossattn_modules(self):
            """ 
            Get ALL attention modules
            """
            modules = module_hooks.get_modules(
                # network_layer_name_filter='middle_block',
                module_name_filter='CrossAttention'
            )
            # regex expression that matches any of the 3 following
            # input_blocks_(5-9)
            # middle_block_xxx
            # output_blocks_(3-5)
            # regexp = re.compile(r'.*input_blocks_[789]_.+|.*middle_block_.*|.*output_blocks_[0123]_.+')
            # modules = [module for module in modules if regexp.match(module.network_layer_name) is not None]
            return modules

    def process_batch(self, p: StableDiffusionProcessing, active, *args, **kwargs):
        active = getattr(p, 'embeds_active', active)
        if not active:
            return

        initno_params= InitnoParams()
        initno_params.token_count, initno_params.max_length = prompt_utils.get_token_count(p.prompt, p.steps, is_positive=True)

        def on_cfg_denoiser(params: CFGDenoiserParams, initno_params: InitnoParams):
            if initno_params.x_in is None:
                initno_params.x_in = params.x
                initno_params.sigma_in = params.sigma
                initno_params.image_cond_in = params.image_cond
                initno_params.text_cond = params.text_cond
                initno_params.text_uncond = params.text_uncond
                logger.debug("Initializing initial_x with shape %s", params.x.shape)
        
        def on_cfg_denoised(params: CFGDenoisedParams, initno_params: InitnoParams):
            x = params.x
            tensor = initno_params.text_cond
            uncond = initno_params.text_uncond
            sigma_in = initno_params.sigma_in
            image_cond_in = initno_params.image_cond_in
            inner_model = params.inner_model

            if params.sampling_step != 2:
                return

            if initno_params.x_in is None:
                return

            if initno_params.ran_once is True:
                return

            initno_params.ran_once = True

            logger.debug("Initial_x is not None, shape: %s", initno_params.x_in.shape)

            # text_cond and uncond don't change
            cond_in = catenate_conds([tensor, uncond])
            make_condition_dict = get_make_condition_dict_fn(uncond)
            conds = make_condition_dict(cond_in, image_cond_in)

            max_step = 50
            max_round = 5

            all_modules = self.get_all_crossattn_modules()
            cross_attn_modules = [module for module in all_modules if hasattr(module, 'initno_crossattn')]
            self_attn_modules = [module for module in all_modules if hasattr(module, 'initno_selfattn')]

            def joint_loss(SCrossAttn, LKL, lambda1 = 1, lambda3=500):
                 return lambda1 * SCrossAttn + lambda3 * LKL

            def crossattn_loss(crossattn_maps, target_tokens):
                """ Calculate the cross-attn loss"""
                # calculate the max cross-attn score
                scores = [torch.max(crossattn_maps[:, i]) for i in range(crossattn_maps.size(-1))]
                return 1.0 - torch.min(torch.tensor(scores))

            def kl_divergence(mu, sigma):
                p = Normal(mu, sigma)
                p = Normal(mu, sigma)
                q = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
                kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

                return kl_loss_fn(p, q)
            
            def self_attn_loss(As, Y):
                return 1.0
            
            def get_crossattn_maps():
                target_size = 256
                crossattn_maps = []

                attn_maps = []
                for module in cross_attn_modules:
                    attnmap = module.initno_crossattn
                    attnmap = attnmap.transpose(-1, -2)
                    if attnmap.size(2) != target_size:
                        attnmap = F.interpolate(attnmap, scale_factor=256/attnmap.size(2), mode='nearest')
                    attnmap = attnmap.transpose(-1, -2)
                    attn_maps.append(attnmap)

                crossattn_maps = torch.stack(attn_maps, dim=1) # b n h t 
                batch_size, num_maps, hw, tokens = crossattn_maps.shape
                crossattn_maps = crossattn_maps.view(batch_size * num_maps, hw, tokens)
                crossattn_maps = crossattn_maps[:, :, target_tokens] # b*n hw t


                # select the target tokens

                # average over all maps
                crossattn_maps = crossattn_maps.mean(dim=0) # hw t

                return crossattn_maps.to(shared.device)

            def evaluate_model(x_in):
                x_out = inner_model(x_in, sigma_in, cond=conds)
                crossattn_maps = get_crossattn_maps()
                selfattn_maps = None
                return x_out, crossattn_maps, selfattn_maps

            cond_token_count = uncond.shape[1]
            prompt_token_count = initno_params.token_count

            start_token = 1
            end_token = min(cond_token_count, prompt_token_count+1)

            target_tokens = list(range(start_token, end_token))

            kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

            with torch.enable_grad():
                torch.autograd.set_detect_anomaly(True)

                for round in range(max_round):
                    x = params.x
                    noise_mean = torch.zeros_like(x, requires_grad=True).to(shared.device)
                    noise_std = torch.ones_like(x, requires_grad=True).to(shared.device)

                    optim = torch.optim.Adam([noise_mean, noise_std], lr=1e-2)

                    for step in range(max_step):
                        x_in = noise_mean + noise_std * x
                        n, c, h, w = x_in.shape

                        x_out, Ac, As = evaluate_model(x_in)

                        loss_crossattn = crossattn_loss(Ac, target_tokens)

                        if loss_crossattn < 0.2:
                            params.x = x_in
                            return

                        #loss_kl = kl_divergence(x , noise_std**2)
                        # log_p = torch.log_softmax(x_out.view(n, c, h*w), dim=-1)
                        # q = torch.ones_like(log_p) / (h * w)
                        # loss_kl = kl_loss_fn(log_p, q)

                        # total loss
                        loss = loss_crossattn
                        loss.requires_grad = True

                        optim.zero_grad()
                        loss.backward(retain_graph=True)
                        optim.step()

            
        script_callbacks.on_cfg_denoiser(lambda params: on_cfg_denoiser(params, initno_params))
        script_callbacks.on_cfg_denoised(lambda params: on_cfg_denoised(params, initno_params))

    def optimize_noise(self, x, t_c=0.2, t_s=0.3):
        """ Optimize noise for a given image.
        Arguments:
            x: input initial latent
            t_c: response score threshold for cross attn
            t_s: response score threshold for self attn
        """
        max_step = 50
        max_round = 5
        noise_mean = torch.zeros_like(x)
        noise_std = torch.ones_like(x)

        optim = torch.optim.Adam([noise_mean, noise_std], lr=0.01)
        for round in range(max_round):
            for step in range(max_step):
                    pass

    def postprocess_batch(self, p, *args, **kwargs):
        pass
    
    def unhook_callbacks(self) -> None:
        script_callbacks.remove_current_script_callbacks()
        crossattn_modules = self.get_all_crossattn_modules()
        for module in crossattn_modules:
            module_hooks.remove_module_forward_hook(module.to_q, 'initno_to_q')
            module_hooks.remove_module_forward_hook(module.to_k, 'initno_to_k')
            module_hooks.remove_module_forward_hook(module, 'initno_cross_attn_hook')
            module_hooks.remove_module_forward_hook(module, 'initno_self_attn_hook')
            module_hooks.modules_remove_field(module, 'initno_selfattn')
            module_hooks.modules_remove_field(module, 'initno_crossattn')

    def get_xyz_axis_options(self) -> dict:
        return {}

# from modules/sd_samplers_cfg_denoiser.py:187-195
def get_make_condition_dict_fn(text_uncond):
        if shared.sd_model.model.conditioning_key == "crossattn-adm":
                make_condition_dict = lambda c_crossattn, c_adm: {"c_crossattn": [c_crossattn], "c_adm": c_adm}
        else:
                if isinstance(text_uncond, dict):
                        make_condition_dict = lambda c_crossattn, c_concat: {**c_crossattn, "c_concat": [c_concat]}
                else:
                        make_condition_dict = lambda c_crossattn, c_concat: {"c_crossattn": [c_crossattn], "c_concat": [c_concat]}
        return make_condition_dict

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
        attn_probs = attn_probs.softmax(dim=-1).to(device=shared.device, dtype=to_q_map.dtype)

        ## avoid nan by converting to float32 and subtracting max 
        # attn_probs = attn_probs.to(dtype=torch.float32) #
        # attn_probs -= torch.max(attn_probs)

        # torch.exp(attn_probs, out = attn_probs)
        # summed = attn_probs.sum(dim=-1, keepdim=True, dtype=torch.float32) + torch.finfo(torch.float32).eps
        # attn_probs /= summed

        # attn_probs = attn_probs.to(dtype=dtype)

        return attn_probs
