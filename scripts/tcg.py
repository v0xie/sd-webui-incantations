import os, sys
import logging
import numpy as np
from PIL import Image
import gradio as gr
import torch
import random
import torch.nn.functional as F
from einops import rearrange

if os.environ.get('INCANT_DEBUG', None):
    # suppress excess logging
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

if __name__ == '__main__' and os.environ.get('INCANT_DEBUG', None):
    sys.path.append(f'{os.getcwd()}')
    sys.path.append(f'{os.getcwd()}/extensions/sd-webui-incantations')
else:
    from scripts.incant_utils import module_hooks, plot_tools, prompt_utils
    from modules import shared, scripts, script_callbacks

from scripts.incant_utils import plot_tools
from scripts.ui_wrapper import UIWrapper


logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


"""
WIP Implementation of https://arxiv.org/abs/2404.11824
Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""


class TCGExtensionScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = [
            "tcg_active",
            "tcg_strength",
            "tcg_f_margin",
            "tcg_f_repl",
            "tcg_theta",
            "tcg_attn_threshold",
            "tcg_sharpness",
            "tcg_selfguidance_scale",
        ]

    def title(self) -> str:
        raise 'TCG [arXiv:2404.11824]'
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion('TCG', open=True):
            with gr.Row():
                image_mask = gr.Image(type='pil', image_mode='L', label="Mask", interactive=False, height = 256)
            with gr.Row():
                generate_mask = gr.Button("Generate Mask", elem_id="tcg_btn_generate_mask")
            with gr.Row():
                with gr.Column():
                    left = gr.Slider(label="Left", value=0.2, minimum=0.0, maximum=1.0, step=0.05, elem_id="tcg_mask_left")
                with gr.Column():
                    with gr.Row():
                        top = gr.Slider(label="Top", value=0.3, minimum=0.0, maximum=1.0, step=0.05, elem_id="tcg_mask_top")
                    with gr.Row():
                        bottom = gr.Slider(label="Bottom", value=0.7, minimum=0.0, maximum=1.0, step=0.05, elem_id="tcg_mask_bottom")
                with gr.Column():
                    right = gr.Slider(label="Right", value=0.5, minimum=0.0, maximum=1.0, step=0.05, elem_id="tcg_mask_right")
            with gr.Row():
                active = gr.Checkbox(label="Active", value=True, elem_id="tcg_active")
                strength = gr.Slider(label="Strength", value=1.0, minimum=-5.0, maximum=5.0, step=0.1, elem_id="tcg_strength")
                f_margin = gr.Slider(label="Margin Force", value=1.0, minimum=-2.0, maximum=2.0, step=0.1, elem_id="tcg_f_margin")
                f_repl = gr.Slider(label="Repulsion Force", value=1.0, minimum=-2.0, maximum=2.0, step=0.1, elem_id="tcg_f_repl")
                theta = gr.Slider(label="Conflict Threshold", value=0.01, minimum=0.0, maximum=1.0, step=0.001, elem_id="tcg_theta")
                threshold = gr.Slider(label="Soft Threshold", value=0.5, minimum=0.0, maximum=1.0, step=0.01, elem_id="tcg_attn_threshold")
                sharpness = gr.Slider(label="Threshold Sharpness", value=10.0, minimum=0.1, maximum=20.0, step=0.1, elem_id="tcg_sharpness")
                selfguidance_scale = gr.Slider(label="Self-Guidance Scale", value=1.0, minimum=-2.0, maximum=2.0, step=0.1, elem_id="tcg_selfguidance_scale")
            with gr.Row():
                start_step = gr.Slider(label="Start Step", value=0, minimum=0, maximum=100, step=1, elem_id="tcg_start_step")
                end_step = gr.Slider(label="End Step", value=0, minimum=1, maximum=100, step=1, elem_id="tcg_end_step")
            opts = [active, strength, f_margin, f_repl, theta, threshold, sharpness, selfguidance_scale, image_mask, start_step, end_step]
            for opt in opts:
                opt.do_not_save_to_config = True

            generate_mask.click(
                create_mask,
                inputs = [left, right, top, bottom],
                outputs = [image_mask]
            )
            
            return opts
    
    def get_modules(self):
        return module_hooks.get_modules( module_name_filter='CrossAttention')
    
    def before_process_batch(self, p, active, strength, f_margin, f_repl, theta, threshold, sharpness, selfguidance_scale, image_mask, start_step, end_step, *args, **kwargs):
        self.unhook_callbacks()
        active = getattr(p, 'tcg_active', active)
        if not active:
            return
        strength = getattr(p, 'tcg_strength', strength)
        f_margin = getattr(p, 'tcg_f_margin', f_margin)
        f_repl = getattr(p, 'tcg_f_repl', f_repl)
        theta = getattr(p, 'tcg_theta', theta)
        threshold= getattr(p, 'tcg_attn_threshold', threshold)
        sharpness = getattr(p, 'tcg_sharpness', sharpness)
        selfguidance_scale = getattr(p, 'tcg_selfguidance_scale', selfguidance_scale)
        start_step = getattr(p, 'tcg_start_step', start_step)
        end_step = getattr(p, 'tcg_end_step', end_step)
        
        batch_size = p.batch_size
        height, width = p.height, p.width
        hw = height * width

        setattr(p, 'tcg_current_step', 0)

        token_count, max_length = prompt_utils.get_token_count(p.prompt, p.steps, is_positive=True)
        min_idx = 1
        max_idx = token_count+1
        token_indices = list(range(min_idx, max_idx))

        def tcg_forward_hook(module, input, kwargs, output):
            current_step = module.tcg_current_step
            if not start_step <= current_step <= end_step:
                return
            # calc attn scores
            q_map = module.tcg_to_q_map # B, HW, inner_dim
            k_map = module.tcg_to_k_map # B, C, inner_dim
            v_map = module.tcg_to_v_map # B, C, inner_dim

            # q_map = prepare_attn_map(q_map, module.heads)
            # k_map = prepare_attn_map(k_map, module.heads)
            # v_map = prepare_attn_map(v_map, module.heads)

            attn_scores = q_map @ k_map.transpose(-1, -2)
            attn_scores *= module.scale
            #channel_dim = q_map.shape[-1]
            #attn_scores /= (channel_dim ** 0.5)
            attn_scores = attn_scores.softmax(dim=-1).to(device=shared.device, dtype=q_map.dtype)

            # select k tokens
            # k_map = k_map.transpose(-1, -2)[..., module.tcg_token_indices].transpose(-1,-2)
            #attn_scores = get_attention_scores(q_map, k_map, dtype=q_map.dtype) # (2*B, H*W, C)
            attn_scores = attn_scores.to(torch.float32)
            B, HW, C = attn_scores.shape

            downscale_h = round((HW * (height / width)) ** 0.5)
            # attn_scores = attn_scores.view(2, attn_scores.size(0)//2, downscale_h, HW//downscale_h, attn_scores.size(-1)).mean(dim=0) # (2*B, HW, C) -> (B, H, W, C)
            attn_scores = attn_scores.view(attn_scores.size(0), downscale_h, HW//downscale_h, attn_scores.size(-1)) # (2*B, HW, C) -> (B, H, W, C)

            # slice attn map
            attn_map = attn_scores[..., module.tcg_token_indices].detach().clone() # (B, H, W, K) where K is the subset of tokens

            # threshold it
            # also represents object shape
            attn_map = soft_threshold(attn_map, threshold=threshold, sharpness=sharpness) # B H W C

            # self-guidance
            # inner_dims = attn_map.shape[1:-1]
            # attn_map = attn_map.view(attn_map.size(0), -1, attn_map.size(-1))

            # shape_sum = torch.sum(attn_map, dim=1) # (B, HW)

            # obj_appearance = shape_sum * attn_map
            # obj_appearance /= shape_sum

            # self_guidance = obj_appearance
            # self_guidance = self_guidance.to(output.dtype)
            # self_guidance_factor = output.detach().clone()
            # self_guidance_factor[..., module.tcg_token_indices] = self_guidance

            # attn_map = attn_map.view(attn_map.size(0), *inner_dims, attn_map.size(-1))

            # region mask
            region_mask = module.tcg_region_mask
            if region_mask.shape[1:3] != attn_map.shape[1:3]:
                region_mask = region_mask.permute(0, 3, 1, 2) # (B, H, W, 1) -> (B, 1, H, W)
                region_mask = F.interpolate(region_mask, size=(attn_map.shape[1:3]), mode='bilinear')
                region_mask = region_mask.permute(0, 2, 3, 1) # (B, 1, H, W) -> (B, H, W, 1)
                module.tcg_region_mask = region_mask

            region_mask_centroid = calculate_centroid(region_mask) # (B, C, 2)

            # detect conflicts and return if none
            conflicts = detect_conflict(attn_map, region_mask, theta) # (B, C)
            if not torch.any(conflicts > 0.01):
                logger.debug("No conflicts detected")
                return

            centroids = calculate_centroid(attn_map) # (B, C, 2)
            #logger.debug(centroids)

            displ_force = displacement_force(attn_map, centroids, region_mask_centroid, f_repl, f_margin, clamp = 10) # B C 2

            # zero out displacement force
            displ_force = displ_force * conflicts.unsqueeze(-1)
            logger.debug("Displacements: %s", displ_force)

            # modify the attn map
            output_attn_map = attn_scores.detach().clone() # B H W C

            modified_attn_map, out_centroids = apply_displacements(output_attn_map[..., module.tcg_token_indices], centroids, displ_force)
            output_attn_map[..., module.tcg_token_indices] = modified_attn_map.squeeze(0)

            output_attn_map = output_attn_map.view(B, -1, C) # B HW C
            # output_attn_map = output_attn_map.permute(0, 2, 1) # B C HW
            output_attn_map = output_attn_map @ v_map

            loss = output - output_attn_map
            loss = normalize_map(loss)
            loss **= 2

            
            output += strength * loss
            # output += selfguidance_scale * self_guidance_factor

            #output = output_attn_map

        def tcg_to_q_hook(module, input, kwargs, output):
                setattr(module.tcg_parent_module[0], 'tcg_to_q_map', output)

        def tcg_to_k_hook(module, input, kwargs, output):
                setattr(module.tcg_parent_module[0], 'tcg_to_k_map', output)

        def tcg_to_v_hook(module, input, kwargs, output):
                setattr(module.tcg_parent_module[0], 'tcg_to_v_map', output)
        
        def cfg_denoised_callback(params: script_callbacks.CFGDenoisedParams):
            for module in self.get_modules():
                setattr(module, 'tcg_current_step', p.tcg_current_step)
            p.tcg_current_step += 1

        script_callbacks.on_cfg_denoised(cfg_denoised_callback)

        mask_H, mask_W = image_mask.size
        temp_region_mask = torch.from_numpy(np.array(image_mask)).unsqueeze(-1).unsqueeze(0).to(torch.float32).to(shared.device) # (1, H, W, 1)
        temp_region_mask = temp_region_mask.repeat(batch_size, 1, 1, 1) # (B, H, W, 1)
        # temp_region_mask = torch.zeros((batch_size, mask_H, mask_W, 1), dtype=torch.float32, device=shared.device) # (B, H, W)
        #temp_region_mask = torch.zeros((batch_size, mask_H, mask_W, 1), dtype=torch.float32, device=shared.device) # (B, H, W)
        #temp_region_mask[0, 1*mask_H//8 : 7*mask_H//8 , 2*mask_W//8 : 5*mask_W//8] = 1.0 # mask the left half ish of the canvas

        for module in self.get_modules():
            if not module.network_layer_name.endswith('attn2'):
                continue
            module_hooks.modules_add_field(module, 'tcg_current_step', 0)
            module_hooks.modules_add_field(module, 'tcg_to_q_map', None)
            module_hooks.modules_add_field(module, 'tcg_to_k_map', None)
            module_hooks.modules_add_field(module, 'tcg_to_v_map', None)
            module_hooks.modules_add_field(module, 'tcg_region_mask', temp_region_mask)
            module_hooks.modules_add_field(module, 'tcg_token_indices', torch.tensor(token_indices, dtype=torch.int32, device=shared.device))
            module_hooks.modules_add_field(module.to_q, 'tcg_parent_module', [module])
            module_hooks.modules_add_field(module.to_k, 'tcg_parent_module', [module])
            module_hooks.modules_add_field(module.to_v, 'tcg_parent_module', [module])
            module_hooks.module_add_forward_hook(module.to_q, tcg_to_q_hook, with_kwargs=True)
            module_hooks.module_add_forward_hook(module.to_k, tcg_to_k_hook, with_kwargs=True)
            module_hooks.module_add_forward_hook(module.to_v, tcg_to_v_hook, with_kwargs=True)
            module_hooks.module_add_forward_hook(module, tcg_forward_hook, with_kwargs=True)

    def postprocess_batch(self, p, *args, **kwargs):
        self.unhook_callbacks()
    
    def unhook_callbacks(self) -> None:
        script_callbacks.remove_current_script_callbacks()
        for module in self.get_modules():
            module_hooks.remove_module_forward_hook(module.to_q, 'tcg_to_q_hook')
            module_hooks.remove_module_forward_hook(module.to_k, 'tcg_to_k_hook')
            module_hooks.remove_module_forward_hook(module.to_v, 'tcg_to_v_hook')
            module_hooks.remove_module_forward_hook(module, 'tcg_forward_hook')
            module_hooks.modules_remove_field(module, 'tcg_current_step')
            module_hooks.modules_remove_field(module, 'tcg_to_q_map')
            module_hooks.modules_remove_field(module, 'tcg_to_k_map')
            module_hooks.modules_remove_field(module, 'tcg_to_v_map')
            module_hooks.modules_remove_field(module, 'tcg_region_mask')
            module_hooks.modules_remove_field(module, 'tcg_token_indices')
            module_hooks.modules_remove_field(module.to_q, 'tcg_parent_module')
            module_hooks.modules_remove_field(module.to_k, 'tcg_parent_module')
            module_hooks.modules_remove_field(module.to_v, 'tcg_parent_module')

    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def process_batch(self, p, *args, **kwargs):
        pass

    def get_xyz_axis_options(self) -> dict:
            xyz_grid = [x for x in scripts.scripts_data if x.script_class.__module__ in ("xyz_grid.py", "scripts.xyz_grid")][0].module
            extra_axis_options = {
                    xyz_grid.AxisOption("[TCG] Active", str, tcg_apply_override('tcg_active', boolean=True), choices=xyz_grid.boolean_choice(reverse=True)),
                    xyz_grid.AxisOption("[TCG] Strength", float, tcg_apply_field("tcg_strength")),
                    xyz_grid.AxisOption("[TCG] Repulsion Force", float, tcg_apply_field("tcg_f_repl")),
                    xyz_grid.AxisOption("[TCG] Margin Force", float, tcg_apply_field("tcg_f_margin")),
                    xyz_grid.AxisOption("[TCG] Conflict Threshold", float, tcg_apply_field("tcg_theta")),
                    xyz_grid.AxisOption("[TCG] Soft Threshold", float, tcg_apply_field("tcg_attn_threshold")),
                    xyz_grid.AxisOption("[TCG] Threshold Sharpness", float, tcg_apply_field("tcg_sharpness")),
                    xyz_grid.AxisOption("[TCG] Self-Guidance Scale", float, tcg_apply_field("tcg_selfguidance_scale")),
            }
            return extra_axis_options



debug_coord = lambda x: (round(x[0,0,0].item(),3), round(x[0,0,1].item(), 3))

def displacement_force(attention_map, verts, target_pos, f_rep_strength, f_margin_strength, clamp=0):
    """ Given a set of vertices, calculate the displacement force given by the sum of margin force and repulsive force.
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the force. Shape: (B, H, W, C)
        verts: torch.Tensor - The centroid vertices of the attention map. Shape: (B, C, 2)
        target : torch.Tensor - The vertices of the targets. Shape: (2)
        f_rep_strength: float - The strength of the repulsive force
        f_margin_strength: float - The strength of the margin force
    Returns:
        torch.Tensor - The displacement force for each vertex. Shape: (B, C, 2)
    """
    B, H, W, C = attention_map.shape
    f_clamp = lambda x: x
    if clamp > 0:
        clamp_min = -clamp
        clamp_max = clamp
        f_clamp = lambda x: torch.clamp(x, min=clamp_min, max=clamp_max)

    f_rep = f_clamp(repulsive_force(f_rep_strength, verts, target_pos))
    f_margin = f_clamp(margin_force(f_margin_strength, H, W, verts))

    logger.debug(f"Repulsive force: {debug_coord(f_rep)}, Margin force: {debug_coord(f_margin)}")
    return f_rep + f_margin


def normalize_map(attnmap):
    """ Normalize the attention map over the channel dimension
    Arguments:
        attnmap: torch.Tensor - The attention map to normalize. Shape: (B, HW, C)
    Returns:
        torch.Tensor - The attention map normalized to (0, 1). Shape: (B, HW, C)
    """
    flattened_attnmap = attnmap.transpose(-1, -2)
    min_val = torch.min(flattened_attnmap, dim=-1).values.unsqueeze(-1) # (B, C, 1)
    max_val = torch.max(flattened_attnmap, dim=-1).values.unsqueeze(-1) # (B, C, 1)
    normalized_attn = (flattened_attnmap - min_val) / ((max_val - min_val) + torch.finfo(attnmap.dtype).eps)
    normalized_attn = normalized_attn.transpose(-1, -2)
    return normalized_attn


def soft_threshold(attention_map, threshold=0.5, sharpness=10):
    """ Soft threshold the attention map channels based on the given threshold. Derived from arXiv:2306.00986
    Arguments:
        attention_map: torch.Tensor - The attention map to threshold. Shape: (B, H, W, C)
        threshold: float - The threshold value between 0.0 and 1.0 relative to the minimum/maximum attention value
        sharpness: float - The sharpness of the thresholding function
    Returns:
        torch.Tensor - The attention map thresholded over all C. Shape: (B, H, W, C)
    """
    def _normalize_map(attnmap):
        """ Normalize the attention map over the channel dimension
        Arguments:
            attnmap: torch.Tensor - The attention map to normalize. Shape: (B, H, W, C) or (B, HW, C)
        Returns:
            torch.Tensor - The attention map normalized to (0, 1). Shape: (B, H, W, C)
        """
        B, H, W, C = attnmap.shape
        flattened_attnmap = attnmap.view(attnmap.shape[0], H*W, attnmap.shape[-1]).transpose(-1, -2) # B, C, H*W
        min_val = torch.min(flattened_attnmap, dim=-1).values.unsqueeze(-1) # (B, C, 1)
        max_val = torch.max(flattened_attnmap, dim=-1).values.unsqueeze(-1) # (B, C, 1)
        normalized_attn = (flattened_attnmap - min_val) / ((max_val - min_val) + torch.finfo(attnmap.dtype).eps)
        normalized_attn = normalized_attn.view(B, C, H*W).transpose(-1, -2) # B, H*W, C
        normalized_attn = normalized_attn.view(B, H, W, C)
        return normalized_attn
    threshold = max(0.0, min(1.0, threshold))
    normalized_attn = _normalize_map(attention_map)
    normalized_attn = _normalize_map(torch.sigmoid(sharpness * (normalized_attn - threshold)))
    return normalized_attn


def distances_to_nearest_edges(verts, h, w):
    """ Calculate the distances and direction to the nearest edge bounded by (H, W) for each channel's vertices 
    Arguments:
        verts: torch.Tensor - The vertices. Shape: (B, C, 2), where the last 2 dims are (y, x)
        h: int - The height of the image
        w: int - The width of the image
    Returns:
        torch.Tensor, torch.Tensor:
          - The minimum distance of each vertex to the nearest edge. Shape: (B, C, 1)
          - The direction to the nearest edge. Shape: (B, C, 4, 2), where the last 2 dims are (y, x)
    """
    # y axis is 0!
    y = verts[..., 0] # (B, C, 2)
    x = verts[..., 1] # (B, C, 2)
    B, C, _ = verts.shape
    
    distances = torch.stack([y, h - y, x, w - x], dim=-1) # (B, C, 4)
    
    directions = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]]).view(1, 1, 4, 2).repeat(B, C, 1, 1) # (4, 2) -> (B, C, 4, 2)
    directions = directions.to(verts.device)
    
    return distances, directions


def min_distance_to_nearest_edge(verts, h, w):
    """ Calculate the distances and direction to the nearest edge bounded by (H, W) for each channel's vertices 
    Arguments:
        verts: torch.Tensor - The vertices. Shape: (B, C, 2), where the last 2 dims are (y, x)
        h: int - The height of the image
        w: int - The width of the image
    Returns:
        torch.Tensor, torch.Tensor:
          - The minimum distance of each vertex to the nearest edge. Shape: (B, C)
          - The direction to the nearest edge. Shape: (B, C, 2), where the last 2 dims are (y, x)
    """
    y = verts[..., 0] # y-axis is 0!
    x = verts[..., 1]
    
    # Calculate distances to the edges (y, h-y, x, w-x)
    # y: distance to top edge
    # h - y: distance to bottom edge
    # x: distance to left edge
    # w - x: distance to right edge
    distances = torch.abs(torch.stack([y, h - y, x, w - x], dim=-1))
    
    # Find the minimum distance and the corresponding closest edge
    min_distances, min_indices = distances.min(dim=-1)
    
    # Map edge indices to direction vectors
    directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]]).to(verts.device)
    nearest_edge_dir = directions[min_indices]
    
    return min_distances, nearest_edge_dir


def margin_force(strength, H, W, verts):
    """ Margin force calculation
    Arguments:
        strength: float - The margin force coefficient
        H: float - The height of the image
        W: float - The width of the image
        verts: torch.Tensor - The vertices of the attention map. Shape: (B, C, 2)
    Returns:
        torch.Tensor - The force for each vertex. Shape: (B, C, 2)
    """
    distances, edge_dirs = distances_to_nearest_edges(verts, H, W) # (B, C, 4), (B, C, 4, 2)
    distances = distances.unsqueeze(-1)

    #distances = distances.unsqueeze(-1) # (B, C, 1)
    force_multiplier = -strength / (distances ** 2 + torch.finfo(distances.dtype).eps)
    forces = force_multiplier * edge_dirs # (B, C, 4, 2)
    forces = forces.sum(dim=-2) # (B, C, 2) # sum over the 4 directions to get total force

    return forces

def warping_force(attention_map, verts, displacements, h, w):
    """ Rescales the attention map based on the displacements. Expects a batch size of 1 to operate on all channels at once.
    Arguments:
        attention_map: torch.Tensor - The attention map to update. Shape: (1, H, W, C)
        verts: torch.Tensor - The centroid vertices of the attention map. Shape: (1, C, 2)
        displacements: torch.Tensor - The displacements to apply. Shape: (1, C, 2), where the last 2 dims are the translation by [Y, X]
        h: int - The height of the image
        w: int - The width of the image
    Returns:
        torch.Tensor - The updated attention map. Shape: (B, H, W, C)
    """
    _, H, W, C = attention_map.shape

    old_centroids = verts # (B, C, 2)
    new_centroids = old_centroids + displacements # (B, C, 2)

    # check if new_centroids are out of bounds
    min_bounds = torch.tensor([0, 0], dtype=torch.float32, device=attention_map.device)
    max_bounds = torch.tensor([h-1, w-1], dtype=torch.float32, device=attention_map.device)
    oob_new_centroids = torch.clamp(new_centroids, min_bounds, max_bounds)

    # diferenct between old and new centroids
    correction = oob_new_centroids - new_centroids
    new_centroids = new_centroids + correction

    s_y = (h - 1)/new_centroids[..., 0] # (B, C) 
    s_x = (w - 1)/new_centroids[..., 1] # (B, C) 
    torch.clamp_max(s_y, 1.0, out=s_y)
    torch.clamp_max(s_x, 1.0, out=s_x)
    if torch.any(s_x < 0.99) or torch.any(s_y < 0.99):
        logger.debug(f"Scaling factor: {s_x}, {s_y}")

    # displacements
    o_new = displacements - correction

    # construct affine transformation matrices (sx, 0, delta_x - o_new_x), (0, sy, delta_y - o_new_y)
    theta = torch.tensor([[1, 0, 0],[0, 1, 0]], dtype=torch.float32, device=attention_map.device)
    theta = theta.unsqueeze(0).repeat(C, 1, 1)
    theta[:, 0, 0] = s_x
    theta[:, 1, 1] = s_y
    theta[:, 0, 2] = o_new[..., 1] / w # X
    theta[:, 1, 2] = o_new[..., 0] / h # Y

    # apply the affine transformation
    grid = F.affine_grid(theta, [C, 1, H, W], align_corners=False) # (C, H, W, 2)

    attention_map = attention_map.permute(3, 0, 1, 2) # (B, H, W, C) -> (C, B, H, W)
    attention_map = attention_map.to(torch.float32)
    out_attn_map = F.grid_sample(attention_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    attention_map = attention_map.permute(1, 2, 3, 0) # (C, B, H, W) -> (B, H, W, C)

    out_attn_map = out_attn_map.permute(1, 2, 3, 0) # (C, B, H, W) -> (B, H, W, C)

    # rescale centroids to pixel space
    return out_attn_map, new_centroids



def repulsive_force(strength, pos_vertex, pos_target):
    """ Repulsive force repels the vertices in the direction away from the target 
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the force. Shape: (B, H, W, C)
        strength: float - The global force coefficient
        pos_vertex: torch.Tensor - The position of the vertex. Shape: (B, C, 2)
        pos_target: torch.Tensor - The position of the target. Shape: (2)
    Returns:
        torch.Tensor - The force away from the target. Shape: (B, C, 2)
    """
    d_pos = pos_vertex - pos_target # (B, C, 2)
    d_pos_norm = d_pos.norm() + torch.finfo(d_pos.dtype).eps # normalize the direction
    #d_pos_norm = d_pos.norm(dim=-1, keepdim=True) + torch.finfo(d_pos.dtype).eps # normalize the direction
    d_pos = d_pos / d_pos_norm
    # d_pos /= d_pos_norm
    force = -(strength ** 2)
    return force / d_pos


def multi_target_force(attention_map, omega, xi, pos_vertex, pos_target):
    """ Multi-target force calculation
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the force. Shape: (B, H, W, C)
        omega: torch.tensor - Coefficients for balancing forces amongst j targets
        xi: float - The global force coefficient
        pos_vertex: torch.Tensor - The position of the vertex. Shape: (B, C, 2)
        pos_target: torch.Tensor - The position of the target. Shape: (B, C, 2)
    Returns:
        torch.Tensor - The multi-target force. Shape: (B, C, 2)
    """
    force = -xi ** 2
    pass


def calculate_region(attention_map):
    """ Given an attention map of shape [B, H, W, C], calculate a bounding box over each C
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the bounding box. Shape: (B, H, W, C)
    Returns:
        torch.Tensor - The bounding box of the region. Shape: (B, C, 4), where the last 4 dims are (y, x, a, b)
        y, x: The top left corner of the bounding box
        a, b: The height and width of the bounding box
    """
    B, H, W, C = attention_map.shape
    # Calculate the sum of attention map along the height and width dimensions
    sum_map = attention_map.sum(dim=(1, 2)) # (B, C)
    # Find the indices of the maximum attention value for each channel
    max_indices = sum_map.argmax(dim=1, keepdim=True) # (B, C)
    # Initialize the bounding box tensor
    bounding_box = torch.zeros((B, C, 4), dtype=torch.int32, device=attention_map.device)
    # Iterate over each channel
    for batch_idx in range(B):
        for channel_idx in range(C):
            # Calculate the row and column indices of the maximum attention value
            row_index = max_indices[batch_idx, channel_idx] // W
            col_index = max_indices[batch_idx, channel_idx] % W
            # Calculate the top left corner coordinates of the bounding box
            y = max(0, row_index - 1)
            x = max(0, col_index - 1)
            # Calculate the height and width of the bounding box
            a = min(H - y, row_index + 2) - y
            b = min(W - x, col_index + 2) - x
            # Store the bounding box coordinates in the tensor
            bounding_box[batch_idx, channel_idx] = torch.tensor([y, x, a, b])
    return bounding_box


def calculate_centroid(attention_map):
    """ Calculate the centroid of the attention map 
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the centroid. Shape: (B, H, W, C)
    Returns:
        torch.Tensor - The centroid of the attention map. Shape: (B, C, 2), where the last 2 dims are (y, x)
    """
    # necessary to avoid inf
    attention_map = attention_map.to(torch.float32)
    B, H, W, C = attention_map.shape

    # Create tensors of the y and x coordinates
    y_coords = torch.arange(H, dtype=attention_map.dtype, device=attention_map.device).view(1, H, 1, 1)
    x_coords = torch.arange(W, dtype=attention_map.dtype, device=attention_map.device).view(1, 1, W, 1)

    # Calculate the weighted sums of the coordinates
    weighted_sum_y = torch.sum(y_coords * attention_map, dim=[1, 2])
    weighted_sum_x = torch.sum(x_coords * attention_map, dim=[1, 2])

    # Calculate the total weights
    total_weights = torch.sum(attention_map, dim=[1, 2]) + torch.finfo(attention_map.dtype).eps

    # Calculate the centroids
    centroid_y = weighted_sum_y / total_weights
    centroid_x = weighted_sum_x / total_weights

    # Combine x and y centroids
    centroids = torch.stack([centroid_y, centroid_x], dim=-1) 
    
    return centroids


def detect_conflict(attention_map, region, theta):
    """
    Detect conflict in an attention map with respect to a designated region in PyTorch.
    Parameters:
    attention_map (torch.Tensor): Attention map of shape (B, H, W, K).
    region (torch.Tensor): Binary mask of shape (B, H, W, 1) indicating the region of interest.
    theta (float): Threshold value.
    Returns:
    torch.Tensor: Conflict detection result of shape (B, K), with values 0 or 1 indicating conflict between tokens and the region.
    """
    # Ensure region is the same shape as the spatial dimensions of attention_map
    assert region.shape[1:3] == attention_map.shape[1:3], "Region mask must match spatial dimensions of attention map"
    # Calculate the mean attention within the region
    #region = region.unsqueeze(-1) # Add channel dimension: (B, H, W) -> (B, H, W, 1)
    # HACK: fixme
    if region.dim() != attention_map.dim():
        attention_in_region = attention_map * region.unsqueeze(-1) # Element-wise multiplication
    else:
        attention_in_region = attention_map * region
    #mean_attention_in_region = attention_in_region[attention_in_region > 0] 
    mean_attention_in_region = torch.sum(attention_in_region, dim=(1, 2)) / torch.sum(region, dim=(1, 2)) # Mean over (H, W)
    # Compare with threshold theta
    conflict = (mean_attention_in_region > theta).float() # Convert boolean to float (0 or 1)
    return conflict


def translate_image_2d(image, tyx):
    """
    Translate an image tensor by (ty, tx).
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    Parameters:
    - image: The image tensor of shape (B, C, H, W) where B = 1
    - tyx: The translation along y and x-axis for each channel (C, 2), where the last 2 dims are the translation by [Y, X]

    Returns:
    - Translated image tensor
    """

    B, C, H, W = image.size()

    # swap B and C
    image = image.transpose(0, 1)  # (C, B, H, W)

    # grid bounds, not doing this means losing information at the edges at low resolution
    # hack to prevent out of boudns when align_corners is false
    # pos = 1 - 1e-3 # hack to prevent out of bounds
    # neg = -pos
    pos, neg = 1, -1

    # Create an grid matrix for the translation
    # (-1, -1) is left top pixel, (1, 1) is right bottom pixel
    h_dim = torch.linspace(neg, pos, H, device=image.device, dtype=image.dtype) # height dim from [-1 (top) to 1 (bottom)]
    w_dim = torch.linspace(neg, pos, W, device=image.device, dtype=image.dtype) # width dim to [-1 (left) to 1 (right)]

    h_dim = h_dim.view(1, H, 1).repeat(C, 1, W)
    w_dim = w_dim.view(1, 1, W).repeat(C, H, 1)

    # translate each dim by the displacements
    ty, tx = tyx[..., 0], tyx[..., 1] # C, C
    h_dim = h_dim + ty.view(C, 1, 1)
    w_dim = w_dim + tx.view(C, 1, 1)

    #h_dim = h_dim.unsqueeze(dim=-1).repeat(1, W, 1) # (C, H, W)
    #w_dim = w_dim.unsqueeze(dim=1).repeat(1, 1, H) # (C, H, W)

    #c_dim = c_dim.unsqueeze(-1)
    h_dim = h_dim.unsqueeze(-1) # (C, H, W, 1)
    w_dim = w_dim.unsqueeze(-1) # (C, H, W, 1)

    # Create 4D grid for 5D input
    grid = torch.cat([h_dim, w_dim], dim=-1) # (C, H, W, 2)

    # Apply the grid to the image using grid_sample
    translated_image = F.grid_sample(image, grid, mode='bicubic', padding_mode='zeros', align_corners=True) # C N H W

    return translated_image.transpose(0, 1) # N C H W


### TODO: do this
def translate_image(image, tx, ty):
    """
    Translate an image tensor by (tx, ty).
    https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html

    Parameters:
    - image: The image tensor of shape (B, C, H, W)
    - tx: The translation along the x-axis (B, C, 2)
    - ty: The translation along the y-axis (B, C, 2)

    Returns:
    - Translated image tensor
    """

    #image = image.unsqueeze(dim=1)

    B, C, H, W = image.size()

    # Create an grid matrix for the translation
    if C > 1:
        c_dim = torch.linspace(-1, 1, C, device=image.device, dtype=image.dtype) # channel dim from [-1 to 1]
    else:
        c_dim = torch.tensor([0], device=image.device, dtype=image.dtype)
    h_dim = torch.linspace(-1, 1, H, device=image.device, dtype=image.dtype) # height dim from [-1 to 1]
    w_dim = torch.linspace(-1, 1, W, device=image.device, dtype=image.dtype) # width dim to [-1 to 1]

    c_dim = c_dim.view(C, 1, 1).repeat(1, H, W)
    h_dim = h_dim.view(1, H, 1).repeat(C, 1, W)
    w_dim = w_dim.view(1, 1, W).repeat(1, H, 1)

    # translate each dim by the displacements
    h_dim = h_dim + ty.squeeze(0).view(C, 1, 1)
    w_dim = w_dim + tx.squeeze(0).view(C, 1, 1)

    c_dim = c_dim.unsqueeze(-1)
    h_dim = h_dim.unsqueeze(-1)
    w_dim = w_dim.unsqueeze(-1)

    # Create 4D grid for 5D input
    grid = torch.cat([c_dim, h_dim, w_dim], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1) # (B, C, H, W, 3)

    image = image.unsqueeze(dim=1) # (B, 1, C, H, W)

    # Apply the grid to the image using grid_sample
    translated_image = F.grid_sample(image, grid, mode='nearest', padding_mode='zeros', align_corners=True)

    return translated_image.squeeze(1)


def apply_displacements(attention_map, verts, displacements):
    """ Update the attention map based on the displacements.
    The attention map is updated by displacing the attention values based on the displacements. 
    - Areas that are displaced out of the attention map are discarded.
    - Areas that are displaced into the attention map are initialized with zeros.
    Arguments:
        attention_map: torch.Tensor - The attention map to update. Shape: (B, H, W, C)
        verts: torch.Tensor - The centroid vertices of the attention map. Shape: (B, C, 2)
        displacements: torch.Tensor - The displacements to apply in pixel space. Shape: (B, C, 2), where the last 2 dims are the translation by [Y, X]
    Returns:
        torch.Tensor - The updated attention map. Shape: (B, H, W, C)
    """
    B, H, W, C = attention_map.shape
    out_attn_map = attention_map.detach().clone()
    out_verts = verts.detach().clone()
    # apply displacements
    for batch_idx in range(B):
        out_attn_map[batch_idx], out_verts[batch_idx] = warping_force(attention_map[batch_idx].unsqueeze(0), verts[batch_idx].unsqueeze(0), displacements[batch_idx], H, W)
        out_attn_map[batch_idx] = out_attn_map[batch_idx].squeeze(0)
#        out_attn_map[batch_idx] = translate_image_2d(attention_map[batch_idx].unsqueeze(0), displacements[batch_idx]).squeeze(0)
#
    #out_attn_map = out_attn_map.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
    return out_attn_map, out_verts


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
        # attn_probs = attn_scores.softmax(dim=-1).to(device=shared.device, dtype=to_q_map.dtype)

        attn_probs = to_q_map @ to_k_map.transpose(-1, -2)
        channel_dim = to_q_map.shape[-1]
        attn_probs /= (channel_dim ** 0.5)
        attn_probs = attn_probs.softmax(dim=-1).to(device=shared.device, dtype=to_q_map.dtype)

        # # avoid nan by converting to float32 and subtracting max 
        # attn_probs = attn_probs.to(dtype=torch.float32) #
        # attn_probs -= torch.max(attn_probs)

        # torch.exp(attn_probs, out = attn_probs)
        # summed = attn_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        # attn_probs /= summed + torch.finfo(torch.float32).eps

        #attn_probs = attn_probs.to(dtype=dtype)

        return attn_probs

#######################
### Debug stuff
def plot_point(image, point, radius=1, color=1.0):
    """ Plot a point on an image tensor
    Arguments:
        image: torch.Tensor - The image tensor to plot the point on. Shape: (B, H, W, C)
        point: tuple - The point to plot (y, x)
        radius: int - The radius of the point
        color: tuple - The color of the point
    Returns:
        torch.Tensor - The image tensor with the point plotted
    """
    y, x = point
    y_min = (y- radius).to(torch.int32)
    y_max = (y + radius + 1).to(torch.int32)
    x_min = (x - radius).to(torch.int32)
    x_max = (x + radius + 1).to(torch.int32)
    image[:, y_min:y_max, x_min:x_max, :] = color


def color_region(image, yx, ab, color=1.0, mode='set'):
    """ Color in a region of an image tensor
    Arguments:
        image: torch.Tensor - The image tensor to plot the point on. Shape: (B, H, W, C)
        yx: (int, int) - The y-coordinate and x-coordinate of the upper left corner of the region 
        ab: (int, int) - The x-coordinate and y-coordinate of the lower right corner of the region
        color: tuple - The color of the region
        mode: str - The mode of coloring. 'set' to set the region to the color, 'add' to add the color to the region
    Returns:
        torch.Tensor - The image tensor with the point plotted
    """
    y, x = yx
    a, b = ab
    if mode == 'set':
        image[:, y:a, x:b, :] = color
    elif mode == 'add':
        image[:, y:a, x:b, :] += color


if __name__ == '__main__':

    tempdir = os.path.join(os.getcwd(), 'temp')
    os.makedirs(tempdir, exist_ok=True)
    img_idx = 0

    # macro for saving to png
    _png = lambda attnmap, name, title: plot_tools.plot_attention_map(
        attnmap[0, :, :, 0],
        save_path=os.path.join(tempdir, f'{name:04}.png'),
        title=f'{title}',
    )

    def _png_batch(attnmap, name, title):
        for batch_idx, _ in enumerate(attnmap):
            for channel_idx in range(attnmap.shape[-1]):
                plot_tools.plot_attention_map(
                    attnmap[batch_idx, :, :, channel_idx],
                    save_path=os.path.join(tempdir, f'batch{batch_idx:04}_{name:04}_{channel_idx:02}.png'),
                    title=f'{title} - Channel {channel_idx}',
                )

    B, H, W, C = 1, 64, 64, 3
    dtype = torch.float16
    device = 'cuda'

    # plotted points as proxies for vertices
    vert_list = [
        [3*H//4, W//2], # (lower middle)
        #[H//4+1, W//4+1],   # (upper left middle quadrant)
        #[H//4+1, W//4+1],   # (upper left middle quadrant)
        #[H//2, W]       # (right middle)
    ]
    target_position = [H//2, W//2]
    # upper left, lower right
    verts = torch.tensor([vert_list], dtype=torch.float16, device='cuda') # B C 2

    # region to represent the target region
    region_yx = [H//4, W//4]
    region_ab = [3*H//4, 3*W//4]

    #region_yx = [H//4, 3*H//4]
    #region_ab = [W//4, 3*W//4]

    # initialize a map with all ones
    attention_map = torch.zeros((B, H, W, C)).to(device, dtype) # B H W C

    # color the target region
    color_region(attention_map, region_yx, region_ab, color=1.0, mode='set')
    _png(attention_map, 0, 'Initial Attn Map')

    attention_map_region = attention_map.detach().clone()
    attention_map_region = torch.zeros((B, H, W, 1)).to(device, dtype) # B H W 1
    color_region(attention_map_region, region_yx, region_ab, color=1.0, mode='set')
    _png(attention_map_region, 1, 'Attn Map Region + Centroid')

    # calculate centroid of region
    centroid = calculate_centroid(attention_map) # (B, C, 2)
    #centroid_points = centroid.squeeze(0).squeeze(0)
    centroid_points = centroid

    # plot region centroid
    for batch_idx in range(B):
        for channel_idx in range(C):
            plot_point(attention_map[batch_idx, ..., channel_idx].unsqueeze(0).unsqueeze(-1), list(centroid_points[batch_idx, channel_idx]), radius=1, color=1)
    #_png(attention_map, 1, 'Attn Map Region + Centroid')

    # plot verts
    attn_map_points = torch.zeros_like(attention_map)
    d_region_yx = [1*H//8, 1*W//8]
    d_region_ab = [6*H//8, 3*W//8]
    # color_region(attn_map_points, d_region_yx, d_region_ab, color=0.5, mode='set')

    c0_region_yx = [12,24]
    c0_region_ab = [36, 48]

    c1_region_yx = [36,36]
    c1_region_ab = [48, 48]

    c2_region_yx = [48, 48]
    c2_region_ab = [63, 63]
    if attn_map_points.shape[-1] > 1:
        color_region(attn_map_points[..., 0].unsqueeze(-1), c0_region_yx, c0_region_ab, color=1, mode='set')
        color_region(attn_map_points[..., 1].unsqueeze(-1), c1_region_yx, c1_region_ab, color=0.5, mode='set')
        color_region(attn_map_points[..., 2].unsqueeze(-1), c2_region_yx, c2_region_ab, color=0.2, mode='set')
    _png_batch(attn_map_points, 3, 'Thresholded Attn Map Points -  Channel 0')
    
    attn_map_points_thresholded = soft_threshold(attn_map_points, threshold=0.5, sharpness=10)
    #_png(attn_map_points_thresholded, 3, 'Thresholded Attn Map Points')
    _png_batch(attn_map_points_thresholded, 4, 'Thresholded Attn Map Points')

    # set areas outside the region to 0
    #attn_map_points = attn_map_points * attention_map
    verts = calculate_centroid(attn_map_points) # (B, C, 2)
    for v in verts.squeeze(0):
        plot_point(attn_map_points, v, radius=1, color=1)

    _png(attn_map_points, 6, 'Attn Map Proxy Map')

    # strengths
    s_margin = 5.0
    s_repl = 1.0

    # test displacement forces 
    displ_zero = torch.tensor([0.0, 0.0], dtype=torch.float16, device='cuda') # B C 2
    #d_zero = torch.tensor([[[0.0, 0]]], dtype=torch.float16, device='cuda') # B C 2
    d_down = torch.tensor([[[0.1, 0]]], dtype=torch.float16, device='cuda') # B C 2

    # simulate displacement forces on our points
    img_idx = 7
    iters = 100
    steps = 5
    for i in range(iters):
        logger.debug('Step %d', i)

        # Check for conflicts between target region and attention map
        theta = 0.01
        conflict_detection = detect_conflict(attn_map_points, attention_map_region, theta) # (B, C)
        logger.debug(f'Conflict Detection: {conflict_detection}')
        if not conflict_detection.any():
            logger.info(f'No conflict detected at iter {i}')
            break

        verts = calculate_centroid(attn_map_points) # (B, C, 2)

        # Displacement forces
        displ_force = displacement_force(attn_map_points, verts, centroid, s_repl, s_margin, clamp = 10) # B C 2

        # zero displacement where conflict is none
        displ_force = displ_force * conflict_detection.unsqueeze(-1)


        if torch.isnan(displ_force).any():
            logger.warning(f'Nan in displ_force at iter {i}')

        # apply displacements and calculate new centroids
        attn_map_points, out_verts = apply_displacements(attn_map_points, verts, displ_force)
        verts = calculate_centroid(attn_map_points) # (B, C, 2)

        # debug output
        copied_map = attn_map_points.detach().clone()
        color_region(copied_map, region_yx, region_ab, color=0.1, mode='add')

        # for v in verts:
        #     plot_point(copied_map, v.squeeze(0), radius=1, color=1.0)

        logger.debug(f'Displacement Force: {debug_coord(displ_force)}, Centroid: {debug_coord(out_verts)}')
        for c in range(C):
            ofs = c * 100 
            #plot_point(copied_map[:c], img_idx+ofs+i+c, radius=1, color=1.0)
            _png(copied_map[..., c].unsqueeze(-1), img_idx+ofs+i, f'Displacement Forces Channel {c} Step {i}')
        #_png(copied_map, img_idx+i, f'Displacement Forces Step {i}')


def create_mask(left, right, top, bottom, width=256, height=256):
    """ Create a PIL.Image mask for the region bounded by the given normalized coordinates 
    Arguments:
        left: float - The left coordinate of the region
        right: float - The right coordinate of the region
        top: float - The top coordinate of the region
        bottom: float - The bottom coordinate of the region
        width: int - The width of the mask
        height: int - The height of the mask
    Returns:
        PIL.Image - The mask image
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    x0, x1 = int(left * width), int(right * width)
    y0, y1 = int(top * height), int(bottom * height)
    x_min, x_max = min(x0, x1), max(x0, x1) 
    y_min, y_max = min(y0, y1), max(y0, y1)

    mask[y_min:y_max, x_min:x_max] = 255
    return Image.fromarray(mask.astype(np.uint8))


# XYZ Plot
# Based on @mcmonkey4eva's XYZ Plot implementation here: https://github.com/mcmonkeyprojects/sd-dynamic-thresholding/blob/master/scripts/dynamic_thresholding.py
def tcg_apply_override(field, boolean: bool = False):
    def fun(p, x, xs):
        if boolean:
            x = True if x.lower() == "true" else False
        setattr(p, field, x)
    return fun

def tcg_apply_field(field):
    def fun(p, x, xs):
        if not hasattr(p, "tcg_active"):
                p.tcg_active = True
        setattr(p, field, x)
    return fun


def prepare_attn_map(to_k_map, heads):
    to_k_map = head_to_batch_dim(to_k_map, heads)
    to_k_map = average_over_head_dim(to_k_map, heads)
    to_k_map = torch.stack([to_k_map[0], to_k_map[0]], dim=0)
    return to_k_map


# based on diffusers/models/attention_processor.py Attention head_to_batch_dim
def head_to_batch_dim(x, heads, out_dim=3):
        head_size = heads
        if x.ndim == 3:

                batch_size, seq_len, dim = x.shape
                extra_dim = 1
        else:
               batch_size, extra_dim, seq_len, dim = x.shape
        x = x.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        x = x.permute(0, 2, 1, 3)
        if out_dim == 3:
               x = x.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)
        return x


# based on diffusers/models/attention_processor.py Attention batch_to_head_dim
def batch_to_head_dim(x, heads):
        head_size = heads
        batch_size, seq_len, dim = x.shape
        x = x.reshape(batch_size // head_size, head_size, seq_len, dim)
        x = x.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return x


def average_over_head_dim(x, heads):
        x = rearrange(x, '(b h) s t -> b h s t', h=heads).mean(1)
        return x

