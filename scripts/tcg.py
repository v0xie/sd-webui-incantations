import os, sys
import logging
import gradio as gr
import torch
import torch.nn.functional as F

if __name__ == '__main__' and os.environ.get('INCANT_DEBUG', None):
    sys.path.append(f'{os.getcwd()}')
    sys.path.append(f'{os.getcwd()}/extensions/sd-webui-incantations')
from scripts.ui_wrapper import UIWrapper
from scripts.incant_utils import module_hooks, plot_tools

"""
WIP Implementation of https://arxiv.org/abs/2404.11824
Author: v0xie
GitHub URL: https://github.com/v0xie/sd-webui-incantations

"""

logger = logging.getLogger(__name__)
if os.environ.get('INCANT_DEBUG', None):
    # suppress excess logging
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

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
    """ Rescales the attention map based on the displacements.
    Arguments:
        attention_map: torch.Tensor - The attention map to update. Shape: (B, H, W, C)
        verts: torch.Tensor - The centroid vertices of the attention map. Shape: (B, C, 2)
        displacements: torch.Tensor - The displacements to apply. Shape: (B, C, 2), where the last 2 dims are the translation by [Y, X]
        h: int - The height of the image
        w: int - The width of the image
    Returns:
        torch.Tensor - The updated attention map. Shape: (B, H, W, C)
    """
    B, H, W, C = attention_map.shape

    # 

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

    # displacements
    o_new = displacements - correction
    #delta_h = old_centroids + displacements[..., 0] - new_centroids[..., 0]# (B, C)
    #delta_w = old_centroids + displacements[..., 1] - new_centroids[..., 1] # (B, C)

    # construct affine transformation matrices (sx, 0, delta_x - o_new_x), (0, sy, delta_y - o_new_y)
    theta = torch.tensor([[1, 0, 0],[0, 1, 0]], dtype=torch.float32, device=attention_map.device)
    theta = theta.unsqueeze(0).repeat(C, 1, 1)
    theta[:, 0, 0] = s_x
    theta[:, 1, 1] = s_y
    #theta[:, 0, 2] = o_new[..., 1] / w # X
    #theta[:, 1, 2] = o_new[..., 0] / h # Y
    theta[:, 0, 2] = o_new[..., 1] / w # X
    theta[:, 1, 2] = o_new[..., 0] / h # Y

    # apply the affine transformation
    grid = F.affine_grid(theta, [B, C, H, W], align_corners=True) # (C, H, W, 2)
    attention_map = attention_map.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
    attention_map = attention_map.to(torch.float32)
    out_attn_map = F.grid_sample(attention_map, grid, mode='bicubic', padding_mode='zeros', align_corners=True)
    attention_map = attention_map.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

    out_attn_map = out_attn_map.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

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
    
    # Get the height and width
    B, H, W, C = attention_map.shape

    # Create a coordinate grid
    y_coords = torch.arange(H).reshape(1, H, 1, 1).expand(B, H, W, C).to(attention_map.device)
    x_coords = torch.arange(W).reshape(1, 1, W, 1).expand(B, H, W, C).to(attention_map.device)
    
    # Flatten the height and width dimensions
    flattened_matrix = attention_map.reshape(B, -1, C)
    y_coords = y_coords.reshape(B, -1, C)
    x_coords = x_coords.reshape(B, -1, C)
    
    # Calculate weighted sums
    total_weight = flattened_matrix.sum(dim=1, keepdim=True)
    centroid_y = (y_coords * flattened_matrix).sum(dim=1, keepdim=True) / total_weight
    centroid_x = (x_coords * flattened_matrix).sum(dim=1, keepdim=True) / total_weight
    
    # Combine x and y centroids
    centroids = torch.cat([centroid_y, centroid_x], dim=-1)
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

        # avoid nan by converting to float32 and subtracting max 
        attn_probs = attn_probs.to(dtype=torch.float32) #
        attn_probs -= torch.max(attn_probs)

        torch.exp(attn_probs, out = attn_probs)
        summed = attn_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)
        attn_probs /= summed

        attn_probs = attn_probs.to(dtype=dtype)

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

    B, H, W, C = 1, 64, 64, 1
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

    # calculate centroid of region
    centroid = calculate_centroid(attention_map) # (B, C, 2)
    centroid_points = centroid.squeeze(0).squeeze(0)

    # plot region centroid
    plot_point(attention_map, centroid_points, radius=1, color=1)
    _png(attention_map, 1, 'Attn Map Region + Centroid')

    # plot verts
    attn_map_points = torch.zeros_like(attention_map)
    d_region_yx = [1*H//8, 1*W//8]
    d_region_ab = [6*H//8, 3*W//8]
    color_region(attn_map_points, d_region_yx, d_region_ab, color=0.5, mode='set')

    # set areas outside the region to 0
    #attn_map_points = attn_map_points * attention_map
    verts = calculate_centroid(attn_map_points) # (B, C, 2)
    for v in verts.squeeze(0):
        plot_point(attn_map_points, v, radius=1, color=1)

    _png(attn_map_points, 2, 'Attn Map Proxy Map')

    # attn_centroid = calculate_centroid(attn_map_points) # (B, C, 2)

    #attn_map_points = attention_map.detach().clone()
    # for v in verts.squeeze(0):
    #     plot_point(attn_map_points, v, radius=1)
    # _png(attn_map_points, 3, 'Points')

    # strengths
    s_margin = 1.0
    s_repl = 1.0

    # displacement forces 
    d_zero = torch.tensor([[[0.0, 0]]], dtype=torch.float16, device='cuda') # B C 2
    d_down = torch.tensor([[[0.1, 0]]], dtype=torch.float16, device='cuda') # B C 2

    # simulate displacement forces on our points
    img_idx = 4
    iters = 100
    steps = 5
    for i in range(iters):
        # copy the map
        bbox_map = calculate_region(attn_map_points) # (B, C, 4)

        displ_force = displacement_force(attn_map_points, verts, centroid, s_repl, s_margin, clamp = 2) # B C 2

        # check for nan
        if torch.isnan(displ_force).any():
            logger.warning(f'Nan in displ_force at iter {i}')

        # fix me
        # displ_force = d_zero

        attn_map_points, out_verts = apply_displacements(attn_map_points, verts, displ_force)

        # new_vert_pos = verts + displ_force
        # new_centroid = calculate_centroid(attn_map_points) # (B, C, 2)


        logger.debug(f'Displacement Force: {debug_coord(displ_force)}, Centroid: {debug_coord(out_verts)}')

        verts = out_verts 

        # copy to put on top visualizations 
        #copied_map = torch.zeros_like(displaced_map).to(device, dtype)
        copied_map = attn_map_points.detach().clone()

        color_region(copied_map, region_yx, region_ab, color=0.1, mode='add')

        for v in verts:
            plot_point(copied_map, v.squeeze(0), radius=1, color=1.0)

        #if i % 10 == 0:
        _png(copied_map, img_idx+i, f'Displacement Forces Step {i}')


    # # conflict detection
    # attention_map = torch.ones(B, H, W, C).to('cuda') # B H W C
    # region = torch.zeros((B, H, W), dtype=torch.float16, device='cuda') # B H W C
    # # set the left half of region to 1
    # region[:, :, :W//2] = 1
    # theta = 0.5 # Example threshold
    # conflict_detection = detect_conflict(attention_map, region, theta)
    # print(conflict_detection)

    # # Create a simple attention map with known values
    # attention_map = torch.zeros((B, H, W, C), device='cuda')  # Shape (batch_size, height, width, channels)
    # attention_map[0, H//2, W//2, 0] = 1.0  # Put all attention on the center
    
    # # Calculate centroids
    # centroids = calculate_centroid(attention_map) # (B, C, 2)
    
    # # Expected centroid is the center of the attention map (2, 2)
    # expected_centroid = torch.tensor([[[H/2, W/2]]], device='cuda')
    
    # # Check if the calculated centroid matches the expected centroid
    # assert torch.allclose(centroids, expected_centroid), f"Expected {expected_centroid}, but got {centroids}"