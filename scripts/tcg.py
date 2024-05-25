import os, sys
import logging
import gradio as gr
import torch

if __name__ == '__main__' and os.environ.get('INCANT_DEBUG', None):
    sys.path.append(f'{os.getcwd()}')
    sys.path.append(f'{os.getcwd()}/extensions/sd-webui-incantations')
from scripts.ui_wrapper import UIWrapper
from scripts.incant_utils import module_hooks

"""
WIP Implementation of https://arxiv.org/abs/2404.11824
Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""

logger = logging.getLogger(__name__)

class TCGExtensionScript(UIWrapper):
    def __init__(self):
        self.infotext_fields: list = []
        self.paste_field_names: list = []

    def title(self) -> str:
        raise 'TCG [arXiv:2404.11824]'
    
    def setup_ui(self, is_img2img) -> list:
        with gr.Accordion('TCG', open=True):
            active = gr.Checkbox(label="Active", value=True)
            opts = [active]
            for opt in opts:
                opt.do_not_save_to_config = True
            return opts
    
    def get_modules(self):
        return module_hooks.get_modules( module_name_filter='CrossAttention')
    
    def before_process_batch(self, p, active, *args, **kwargs):
        self.unhook_callbacks()
        active = getattr(p, 'tcg_active', active)
        if not active:
            return

        def tcg_forward_hook(module, input, kwargs, output):
            pass

        def tcg_to_q_hook(module, input, kwargs, output):
                setattr(module.tcg_parent_module[0], 'tcg_to_q_map', output)

        def tcg_to_k_hook(module, input, kwargs, output):
                setattr(module.tcg_parent_module[0], 'tcg_to_k_map', output)

        for module in self.get_modules():
            if not module.network_layer_name.endswith('attn2'):
                continue
            module_hooks.modules_add_field(module, 'tcg_to_q_map', None)
            module_hooks.modules_add_field(module, 'tcg_to_k_map', None)
            module_hooks.modules_add_field(module.to_q, 'tcg_parent_module', [module])
            module_hooks.modules_add_field(module.to_k, 'tcg_parent_module', [module])
            module_hooks.module_add_forward_hook(module.to_q, tcg_to_q_hook, with_kwargs=True)
            module_hooks.module_add_forward_hook(module.to_k, tcg_to_k_hook, with_kwargs=True)
            module_hooks.module_add_forward_hook(module, tcg_forward_hook, with_kwargs=True)

    def postprocess_batch(self, p, *args, **kwargs):
        self.unhook_callbacks()
    
    def unhook_callbacks(self) -> None:
        for module in self.get_modules():
            module_hooks.remove_module_forward_hook(module.to_q, 'tcg_to_q_hook')
            module_hooks.remove_module_forward_hook(module.to_k, 'tcg_to_k_hook')
            module_hooks.modules_remove_field(module, 'tcg_to_q_map')
            module_hooks.modules_remove_field(module, 'tcg_to_k_map')
            module_hooks.modules_remove_field(module.to_q, 'tcg_parent_module')
            module_hooks.modules_remove_field(module.to_k, 'tcg_parent_module')

    def before_process(self, p, *args, **kwargs):
        pass

    def process(self, p, *args, **kwargs):
        pass

    def process_batch(self, p, *args, **kwargs):
        pass

    def get_xyz_axis_options(self) -> dict:
        return {}


def displacement_force(attention_map, verts, f_rep_strength, f_margin_strength):
    """ Given a set of vertices, calculate the displacement force given by the sum of margin force and repulsive force.
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the force. Shape: (B, H, W, C)
        verts: torch.Tensor - The vertices of the attention map. Shape: (B, C, 2)
        f_rep_strength: float - The strength of the repulsive force
        f_margin_strength: float - The strength of the margin force
    """
    B, H, W, C = attention_map.shape
    f_rep = repulsive_force(f_rep_strength, verts, calculate_centroid(attention_map))
    f_margin = margin_force(f_margin_strength, H, W, verts)
    return f_rep + f_margin


def min_distance_to_nearest_edge(verts, h, w):
    """ Calculate the distances of the vertices from the nearest edge given the height and width of the image 
    Arguments:
        verts: torch.Tensor - The vertices of the attention map. Shape: (B, C, 2)
        h: int - The height of the image
        w: int - The width of the image
    """
    x_coords, y_coords = verts[:, :, 0], verts[:, :, 1] # coordinates
    distances_to_edges = torch.stack([y_coords, h - y_coords, x_coords, w - x_coords], dim=-1) # (B, C, 4)
    min_distances = torch.min(distances_to_edges, dim=-1).values # (B, C)
    return min_distances


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
    min_distances = min_distance_to_nearest_edge(verts, H, W) # (B, C)
    force = -strength / (min_distances ** 2)
    return force


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
    d_pos_norm = d_pos.norm(dim=-1, keepdim=True) # normalize the direction
    d_pos /= d_pos_norm
    force = (-strength) ** 2 
    return force * d_pos


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

    

def calculate_centroid(attention_map):
    """ Calculate the centroid of the attention map 
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the centroid. Shape: (B, H, W, C)
    Returns:
        torch.Tensor - The centroid of the attention map. Shape: (B, C, 2)
    """
    
    # Get the height and width
    B, H, W, C = attention_map.shape

    h_coords = torch.arange(H).view(1, H, 1, 1).to(attention_map.device)
    w_coords = torch.arange(W).view(1, 1, W, 1).to(attention_map.device)
    
    # Sum of attention scores for each channel
    attention_sum = torch.sum(attention_map, dim=(1, 2)) # shape: (B, C)
    
    # Weighted sum of the coordinates
    h_weighted_sum = torch.sum(h_coords * attention_map, dim=(1,2)) # (B, C)
    w_weighted_sum = torch.sum(w_coords * attention_map, dim=(1,2)) # (B, C)
    
    # Calculate the centroids
    centroid_h = h_weighted_sum / attention_sum
    centroid_w = w_weighted_sum / attention_sum
    
    centroids = torch.stack([centroid_h, centroid_w], dim=-1) # (B, C, 2)
    
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
    assert region.shape[1:] == attention_map.shape[1:3], "Region mask must match spatial dimensions of attention map"
    # Calculate the mean attention within the region
    region = region.unsqueeze(-1) # Add channel dimension: (B, H, W) -> (B, H, W, 1)
    attention_in_region = attention_map * region # Element-wise multiplication
    mean_attention_in_region = torch.sum(attention_in_region, dim=(1, 2)) / torch.sum(region, dim=(1, 2)) # Mean over (H, W)
    # Compare with threshold theta
    conflict = (mean_attention_in_region > theta).float() # Convert boolean to float (0 or 1)
    return conflict



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

        # avoid nan by converting to float32 and subtracting max 
        attn_probs = attn_probs.to(dtype=torch.float32) #
        attn_probs -= torch.max(attn_probs)

        torch.exp(attn_probs, out = attn_probs)
        summed = attn_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        attn_probs /= summed

        attn_probs = attn_probs.to(dtype=dtype)

        return attn_probs


if __name__ == '__main__':
    # repulsive force
    B, H, W, C = 1, 64, 64, 1
    verts = torch.tensor([[[8, 8], [16, 48], [31, 31]]], dtype=torch.float16, device='cuda') # B C 2
    target = torch.tensor([[[32, 32]]], dtype=torch.float16, device='cuda') # B 1 2
    r_force = repulsive_force(1, verts, target)




    # conflict detection
    attention_map = torch.ones(B, H, W, C).to('cuda') # B H W C
    region = torch.zeros((B, H, W), dtype=torch.float16, device='cuda') # B H W C
    # set the left half of region to 1
    region[:, :, :W//2] = 1
    theta = 0.5 # Example threshold
    conflict_detection = detect_conflict(attention_map, region, theta)
    print(conflict_detection)

    # Create a simple attention map with known values
    attention_map = torch.zeros((B, H, W, C), device='cuda')  # Shape (batch_size, height, width, channels)
    attention_map[0, H//2, W//2, 0] = 1.0  # Put all attention on the center
    
    # Calculate centroids
    centroids = calculate_centroid(attention_map) # (B, C, 2)
    
    # Expected centroid is the center of the attention map (2, 2)
    expected_centroid = torch.tensor([[[H/2, W/2]]], device='cuda')
    
    # Check if the calculated centroid matches the expected centroid
    assert torch.allclose(centroids, expected_centroid), f"Expected {expected_centroid}, but got {centroids}"