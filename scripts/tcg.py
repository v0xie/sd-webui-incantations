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
    

def calculate_centroid(attention_map):
    """ Calculate the centroid of the attention map 
    Arguments:
        attention_map: torch.Tensor - The attention map to calculate the centroid. Shape: (batch_size, height, width, channels)
    Returns:
        torch.Tensor - The centroid of the attention map. Shape: (batch_size, 2, channels)
    """
    
    # Get the height and width
    batch_size, height, width, channels = attention_map.shape
    
    # Create a mesh grid of height and width coordinates
    h_coords = torch.arange(height).unsqueeze(1).expand(height, width).to(attention_map.device)
    w_coords = torch.arange(width).unsqueeze(0).expand(height, width).to(attention_map.device)
    
    # Flatten the coordinates to apply the sum
    h_coords = h_coords.reshape(-1)
    w_coords = w_coords.reshape(-1)
    
    # Flatten the attention_map for easier manipulation
    attention_map_flat = attention_map.view(batch_size, -1, channels)
    
    # Sum of attention scores for each channel
    attention_sum = attention_map_flat.sum(dim=1, keepdim=True) + 1e-10  # Add small value to avoid division by zero
    
    # Weighted sum of the coordinates
    h_weighted_sum = (h_coords.unsqueeze(0) * attention_map_flat).sum(dim=1)
    w_weighted_sum = (w_coords.unsqueeze(0) * attention_map_flat).sum(dim=1)
    
    # Calculate the centroids
    centroid_h = h_weighted_sum / attention_sum
    centroid_w = w_weighted_sum / attention_sum
    
    # Combine the centroids into a single tensor of shape (batch_size, 2, channels)
    centroids = torch.stack([centroid_h, centroid_w], dim=1)
    
    return centroids


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
    # Create a simple attention map with known values
    attention_map = torch.zeros((1, 5, 5, 1))  # Shape (batch_size, height, width, channels)
    attention_map[0, 2, 2, 0] = 1  # Put all attention on the center
    
    # Calculate centroids
    centroids = calculate_centroid(attention_map)
    
    # Expected centroid is the center of the attention map (2, 2)
    expected_centroid = torch.tensor([[[2.0], [2.0]]])
    
    # Check if the calculated centroid matches the expected centroid
    assert torch.allclose(centroids, expected_centroid), f"Expected {expected_centroid}, but got {centroids}"
    print("Sanity check passed!")    