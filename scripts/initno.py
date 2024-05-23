import gradio as gr
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur
import logging
import re
from scripts.ui_wrapper import UIWrapper
from torchvision.transforms import ToPILImage
from modules import shared, script_callbacks, shared_state
from modules.processing import StableDiffusionProcessing
from modules.script_callbacks import CFGDenoiserParams, CFGDenoisedParams, AfterCFGCallbackParams
from modules.sd_samplers_cfg_denoiser import catenate_conds
from torch.distributions import Normal

from scripts.incant_utils import module_hooks, prompt_utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def show_image(tensor):
    """ Expects 2/3 dim tensor x, y"""
    to_img = ToPILImage()
    # upres
    tensor = tensor.unsqueeze(0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, scale_factor=16, mode='bilinear')
    tensor = tensor.squeeze(0) # c, h, w
    # repeat channel if missing 1
    if tensor.size(0) == 2:
        tensor = tensor.sum(dim=0)
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        #tensor = torch.cat([tensor, tensor[0].unsqueeze(0)], dim=0)
    to_img(tensor).show()


class InitnoParams():
    def __init__(self):
        self.max_step = None
        self.max_round = None
        self.t_c = None
        self.t_s = None
        self.lr = None
        self.tokens = []
        self.x_in = None
        self.sigma_in = None
        self.image_cond_in = None
        self.text_cond = None
        self.text_uncond = None
        self.token_count = None
        self.max_length = None
        self.ran_once = False
        self.replaced_noise = False
        self.optimized_noise = None

class InitnoScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []
        self.cached_c = [None, None]

    def title(self) -> str:
        return "Initno"

    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion(label="Initno [arXiv:2404.04650]", open=False):
            active = gr.Checkbox(label="Active", default=True, elem_id='embeds_active')
            max_step = gr.Slider(label="Max Steps", value=50, min=1, max=150, default=50, elem_id='max_step')
            max_round = gr.Slider(label="Max Rounds", value=5, min=1, max=15, default=5, elem_id='max_round')
            t_c = gr.Slider(label="Cross Attn Threshold", value=0.2, min=0.1, max=1.0, default=0.2, step=0.01, elem_id='t_c')
            t_s = gr.Slider(label="Self Attn Threshold", value=0.3, min=0.1, max=1.0, default=0.3, step=0.01, elem_id='t_s')
            lr = gr.Slider(label="Learning Rate", value=1e-2, min=1e-4, max=1e1, default=1e-2, step=1e-4, elem_id='lr')
            tokens = gr.Textbox(label="Target Tokens", value="", default="", elem_id='target_tokens')

        params = [active, max_step, max_round, t_c, t_s, lr, tokens]
        for param in params:
            param.do_not_save_to_config = True
        return params

    def get_infotext_fields(self) -> list:
        return self.infotext_fields

    def get_paste_field_names(self) -> list:
        return self.paste_field_names
    
    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def before_process_batch(self, p: StableDiffusionProcessing, active, max_step, max_round, t_c, t_s, lr, tokens, *args, **kwargs):
        self.unhook_callbacks()
        active = getattr(p, 'embeds_active', active)
        if not active:
            return
        
        def initno_to_q(module, input, kwargs, output):
            setattr(module.initno_parent_module[0], 'to_q_map', output)

        def initno_to_k(module, input, kwargs, output):
            setattr(module.initno_parent_module[0], 'to_k_map', output)

        def initno_cross_attn_hook(module, input, kwargs, output):
            indices = module.initno_indices
            q_map = module.to_q_map.detach().clone()
            k_map = module.to_k_map.detach().clone()

            k_map -= k_map[:, 0, :].unsqueeze(dim=1) # subtract sot
            k_map = k_map[:, indices, :] 
            
            attn_probs = get_attention_scores(q_map, k_map, dtype=q_map.dtype)
            setattr(module, 'initno_crossattn', attn_probs)

        def initno_self_attn_hook(module, input, kwargs, output):
            q_map = module.to_q_map.detach().clone()
            attn_probs = get_attention_scores(q_map, q_map, dtype=q_map.dtype)
            setattr(module, 'initno_selfattn', attn_probs)

        token_count, max_length = prompt_utils.get_token_count(p.prompt, p.steps, is_positive=True)
        token_count = min(max(token_count, 1), max_length)
        token_indices = list(range(1, token_count+1))

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
                module_hooks.modules_add_field(module, 'initno_indices', torch.tensor(token_indices).to(shared.device))
        
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

    def process_batch(self, p: StableDiffusionProcessing, active, max_step, max_round, t_c, t_s, lr, tokens, *args, **kwargs):
        active = getattr(p, 'embeds_active', active)
        if not active:
            return

        initno_params= InitnoParams()
        initno_params.max_step = max_step
        initno_params.max_round = max_round
        initno_params.t_c = t_c
        initno_params.t_s = t_s
        initno_params.lr = lr

        initno_params.token_count, initno_params.max_length = prompt_utils.get_token_count(p.prompt, p.steps, is_positive=True)
        if tokens is None or tokens == '' or len(tokens) <= 0:

            token_count, max_length = prompt_utils.get_token_count(p.prompt, p.steps, is_positive=True)
            token_count = min(max(token_count, 1), max_length)
            initno_params.tokens = list(range(1, token_count))
        else:
            initno_tokens = [int(token) for token in tokens.split(",")]
            initno_params.tokens = initno_tokens

        def on_cfg_denoiser(params: CFGDenoiserParams, initno_params: InitnoParams):
            ip = initno_params
            if initno_params.x_in is None:
                initno_params.x_in = params.x
                initno_params.sigma_in = params.sigma
                initno_params.image_cond_in = params.image_cond
                initno_params.text_cond = params.text_cond
                initno_params.text_uncond = params.text_uncond
                logger.debug("Initializing initial_x with shape %s", params.x.shape)
            
            if ip.ran_once is False:
                return
            if ip.replaced_noise is True:
                return
            if ip.optimized_noise is None:
                return

            params.x = ip.optimized_noise
            ip.replaced_noise = True

        
        def on_cfg_denoised(params: CFGDenoisedParams, initno_params: InitnoParams):
            x = params.x
            tensor = initno_params.text_cond
            uncond = initno_params.text_uncond
            sigma_in = initno_params.sigma_in
            image_cond_in = initno_params.image_cond_in
            inner_model = params.inner_model

            if params.sampling_step != 1:
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

            max_step = initno_params.max_step
            max_round = initno_params.max_round
            t_c = initno_params.t_c
            t_s = initno_params.t_s
            lr = initno_params.lr

            all_modules = self.get_all_crossattn_modules()
            cross_attn_modules = [module for module in all_modules if hasattr(module, 'initno_crossattn')]
            self_attn_modules = [module for module in all_modules if hasattr(module, 'initno_selfattn')]

            kernel_size = 3
            sigma = 0.5
            blur = GaussianBlur(kernel_size, sigma).to(shared.device)

            def joint_loss(SCrossAttn, SSelfAttn, LKL, lambda1 = 1, lambda2 = 1, lambda3=500):
                 return lambda1 * SCrossAttn + lambda2 * SSelfAttn + lambda3 * LKL

            def crossattn_loss(crossattn_maps, target_tokens):
                """ Calculate the cross-attn loss"""
                # calculate the max cross-attn score
                max_scores = crossattn_maps.max(dim=1).values
                #max_scores = max_scores[:, target_tokens]

                # # if scores are below a certain threshold they're probably never going to get better so we remove them
                logger.debug("CrossAttn: Max scores: %s", max_scores)
                max_scores = max_scores[max_scores > 0.01]

                if max_scores.size(0) == 0:
                    return 1.0
                
                # zip the max scores and target tokens
                #debug_max_scores = list(zip(max_scores, target_tokens))

                scores = [torch.max(crossattn_maps[:, :, i]) for i in range(crossattn_maps.size(-1))]
                return 1.0 - torch.min(max_scores)

            def self_attention_loss(As, Ac):
                """ 
                As - self attention maps ( b hw hw ), hw = 16*16 = 256
                Ac - cross attention maps ( b hw t ), hw = 16:16, t = target tokens
                """
                # query the Ac map for the maximum cross_attn_value for each token
                As = As.mean(0)
                #As = As.mean(0)
                batch_size, hw, tokens = Ac.shape
                As = As.view(16, 16, As.size(-1))

                h = int(np.sqrt(hw))
                w = hw // h

                ac_max = Ac.view(batch_size, h, w, tokens)[0]

                # find x_i, y_i such that Ac[x_i, y_i, token_idx] is max value for token_idx
                # coords = [(x_i, y_i), (x_i+1, y_i+1), ...]
                coords = []
                #for token_idx in target_tokens:
                for token_idx in range(ac_max.size(-1)):
                    max_idx = torch.argmax(ac_max[:, :, token_idx])
                    x_i, y_i = max_idx // ac_max.size(0), max_idx % ac_max.size(1)
                    coords.append(
                        (
                            x_i.item(), y_i.item()
                        )
                    )
                coords = torch.tensor(coords)

                conflicts = 0
                pairs = 0

                # calcuate conflicts for each pair of tokens
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        x_i, y_i = coords[i]
                        x_j, y_j = coords[j]

                        As_xi_yi = As[x_i, y_i]
                        As_xj_yj = As[x_j, y_j]

                        # calculate f(y_i, y_j)
                        num = torch.sum(torch.min(As_xi_yi, As_xj_yj))
                        denom = torch.sum(As_xi_yi + As_xj_yj + 1e-8)
                        f_yi_yj = num / denom

                        conflicts += f_yi_yj
                        pairs += 1

                SSelfAttn = conflicts / pairs if pairs > 0 else 0
                return SSelfAttn

            def kl_divergence(mu, sigma):
                return 0.5 * torch.sum(mu ** 2 + sigma ** 2 - torch.log(sigma ** 2) - 1)
            
            def resize_attn_map(attnmap, target_size=256):
                attnmap = attnmap.transpose(-1, -2)
                if attnmap.size(2) != target_size:
                    attnmap = F.interpolate(attnmap, scale_factor=256/attnmap.size(2), mode='nearest')
                attnmap = attnmap.transpose(-1, -2)
                return attnmap

            def get_cross_attn_maps(modules, map_name):
                target_size = 256
                attn_maps = [getattr(module, map_name) for module in modules]
                attn_maps = [attn_map for attn_map in attn_maps if attn_map.size(1) == target_size]

                #attn_maps = [
                #    resize_attn_map(
                #            attnmap = getattr(module, map_name),
                #            target_size = target_size
                #        ) for module in modules 
                #    ]

                return prepare_attn_maps(attn_maps, target_tokens)

            def get_self_attn_maps(modules, map_name):
                target_size = 256
                attn_maps = [ module.initno_selfattn for module in modules ]
                attn_maps = [ attnmap for attnmap in attn_maps if attnmap.size(-1) == target_size]
                selfattn_maps = torch.stack(attn_maps, dim=1) # b n h w 
                b, n, h, w = selfattn_maps.shape
                selfattn_maps = selfattn_maps.reshape(b*n, h, w)
                selfattn_maps = smooth_attn_map(selfattn_maps)
                selfattn_maps = selfattn_maps.reshape(b, n, h, w)
                selfattn_maps = selfattn_maps.mean(dim=1) # b h w
                return selfattn_maps.to(shared.device)

            def prepare_attn_maps(attn_maps, Y):
                crossattn_maps = torch.stack(attn_maps, dim=1) # b n hw t 
                crossattn_maps = crossattn_maps.transpose(-1, -2) # b n t hw
                b, n, t, hw = crossattn_maps.shape
                crossattn_maps = crossattn_maps.reshape(b * n, t, 16, 16).transpose(0, 1) # t b*n 16 16
                crossattn_maps = smooth_attn_map(crossattn_maps)
                crossattn_maps = crossattn_maps.transpose(0, 1) # b*n t 16 16
                crossattn_maps = crossattn_maps.reshape(b, n, t, hw)
                crossattn_maps = crossattn_maps.transpose(-1, -2) # b n hw t

                #batch_size, num_maps, hw, tokens = crossattn_maps.shape
                crossattn_maps = crossattn_maps.mean(dim=1)[0].unsqueeze(0) # b hw t
                crossattn_maps = crossattn_maps[:, :, Y] # b hw t

                return crossattn_maps.to(shared.device)
            
            def smooth_attn_map(attn_map):
                # use a gaussian kernel to smooth the attention map
                smooth_attn_map = blur(attn_map)
                return smooth_attn_map



            def evaluate_model(x_in):
                x_out = inner_model(x_in, sigma_in, cond=conds)
                crossattn_maps = get_cross_attn_maps(cross_attn_modules, 'initno_crossattn')
                selfattn_maps = get_self_attn_maps(self_attn_modules, 'initno_selfattn')
                return x_out, crossattn_maps, selfattn_maps

            cond_token_count = uncond.shape[1]
            prompt_token_count = initno_params.token_count

            target_tokens = initno_params.tokens
            if initno_params.tokens is None:
                start_token = 1
                end_token = min(cond_token_count, prompt_token_count+1)
                target_tokens = list(range(start_token, end_token))

            kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

            with torch.enable_grad():
                # torch.autograd.set_detect_anomaly(True)

                noise_pool = []

                for round in range(max_round):
                    logger.debug("Initno: Round %d", round)

                    x = params.x.detach().clone() # assuming x[0] is positive and x[1] is neg

                    n, c, h, w = x.shape

                    #x_pos, x_neg = x[0], x[1] # c h w

                    noise_mean = torch.zeros_like(x, requires_grad=True).to(shared.device)
                    noise_std = torch.ones_like(x, requires_grad=True).to(shared.device)

                    optim = torch.optim.AdamW([noise_mean, noise_std], lr=lr)

                    for step in range(max_step):
                        x_new = noise_mean + noise_std * x# c h w

                        x_in = x_new
                        #x_in = torch.stack([x_new, x_neg], dim=0) # 2 c h w

                        x_out, Ac, As = evaluate_model(x_in)

                        _, c, h, w = x_out.shape

                        loss_crossattn = crossattn_loss(Ac, target_tokens)
                        
                        loss_selfattn = self_attention_loss(As, Ac)

                        if loss_crossattn < t_c and loss_selfattn < t_s:
                            logger.debug("Initno: Optimized noise! - Step: %i/%i, CrossAttn: %f < %f, SelfAttn: %f < %f,", step, max_step, loss_crossattn.item(), t_c, loss_selfattn.item(), t_s)
                            params.x = x_in
                            return

                        #log_p = torch.log_softmax(x_out[0].view(c, h*w), dim=-1) # N( mean, std ** 2)
                        #mu = x_out[0].mean()
                        #sigma = x_out[0].std()
                        #q = torch.ones_like(log_p) / (h * w) # N(0, 1)
                        loss_kl = kl_divergence(noise_mean, noise_std)
                        #loss_kl = -torch.log(noise_std) + (noise_std**2 + noise_mean**2) / 2 - 0.5
                        #loss_kl = kl_loss_fn(log_p, q)

                        # total loss
                        loss = joint_loss(loss_crossattn, loss_selfattn, loss_kl)

                        logger.debug("Initno: Step: %i/%i, Loss:%f, CrossAttn:%f, SelfAttn:%f, KL:%f", step, max_step, loss.item(), loss_crossattn.item(), loss_selfattn.item(), loss_kl.item())
                        optim.zero_grad()
                        loss.backward(retain_graph=True)
                        optim.step()

                        if shared.state.interrupted or shared.state.skipped or shared.state.stopping_generation:
                            return
                    
                    logger.debug("Initno: Failed to optimize noise, next round...")
                    noise_pool.append((loss_crossattn, loss_selfattn, noise_mean.detach().clone(), noise_std.detach().clone()))

                # select the noise with the lowest combined loss
                noise_pool = sorted(noise_pool, key=lambda x: x[0] + x[1])
                loss_crossattn, loss_selfattn, best_noise_mean, best_noise_std = noise_pool[0]

                logger.debug("Initno: Best noise found with crossattn: %f, selfattn: %f", loss_crossattn, loss_selfattn)

                initno_params.optimized_noise = best_noise_mean + best_noise_std * x
                #initno_params.optimized_noise = torch.stack([initno_params.optimized_noise, x_neg], dim=0) # 2 c h w
            
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
            module_hooks.modules_remove_field(module, 'initno_indices')

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

        channel_dims = to_q_map.size(-1)
        attn_probs = to_q_map @ to_k_map.transpose(-1, -2)
        #attn_probs /= channel_dims ** 0.5
        ## attn_probs /= to_q_map.size(-1) ** 0.5
        attn_probs = attn_probs.softmax(dim=-1).to(device=shared.device, dtype=dtype)

        ### avoid nan by converting to float32 and subtracting max 
        #attn_probs = attn_probs.to(dtype=torch.float32) #
        #attn_probs -= torch.max(attn_probs)

        #torch.exp(attn_probs, out = attn_probs)
        #summed = attn_probs.sum(dim=-1, keepdim=True, dtype=torch.float32) + torch.finfo(torch.float32).eps
        #attn_probs /= summed

        #attn_probs = attn_probs.to(dtype=dtype)

        return attn_probs
