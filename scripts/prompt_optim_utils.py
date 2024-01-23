"""
MIT License

Copyright (c) 2022 Yuxin Wen and Neel Jain

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# copied and modified from https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py and https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/f22a1bec01991d94697304443cacbd66e0167e6b/open_clip/model.py 
# commit f22a1be

import random
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from statistics import mean
import copy
import json
from typing import Any, Mapping

import open_clip
import clip

import torch
from scipy.optimize import fmin_l_bfgs_b

from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)

run_args = {
    "prompt_len": 16,
    "iter": 1000,
    "lr": 0.1,
    "weight_decay": 0.1,
    "prompt_bs": 1,
    "print_step": 100,
    "batch_size": 1,
    "print_new_best": True,
    "loss_weight": 1.0,
    "clip_model": "ViT-L/14",
    "pretrained": "openai",
}


def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def nn_project(curr_embeds, embedding_layer, print_hits=False):
    with torch.no_grad():
        bsz,seq_len,emb_dim = curr_embeds.shape
        
        # Using the sentence transformers semantic search which is 
        # a dot product exact kNN search between a set of 
        # query vectors and a corpus of vectors
        curr_embeds = curr_embeds.reshape((-1,emb_dim))
        curr_embeds = normalize_embeddings(curr_embeds) # queries

        embedding_matrix = embedding_layer.weight
        embedding_matrix = normalize_embeddings(embedding_matrix)
        
        hits = semantic_search(curr_embeds, embedding_matrix, 
                                query_chunk_size=curr_embeds.shape[0], 
                                top_k=1,
                                score_function=dot_score)

        if print_hits:
            all_hits = []
            for hit in hits:
                all_hits.append(hit[0]["score"])
            print(f"mean hits:{mean(all_hits)}")
        
        nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device)
        nn_indices = nn_indices.reshape((bsz,seq_len))

        projected_embeds = embedding_layer(nn_indices)

    return projected_embeds, nn_indices


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def decode_ids(input_ids, tokenizer, by_token=False):
    input_ids = input_ids.detach().cpu().numpy()

    texts = []

    if by_token:
        for input_ids_i in input_ids:
            curr_text = []
            for tmp in input_ids_i:
                curr_text.append(tokenizer.decode([tmp]))

            texts.append('|'.join(curr_text))
    else:
        for input_ids_i in input_ids:
            texts.append(tokenizer.decode(input_ids_i))

    return texts


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


def get_target_feature(model, preprocess, tokenizer_funct, device, target_images=None, target_prompts=None):
    if target_images is not None:
        with torch.no_grad():
            curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
            curr_images = torch.concatenate(curr_images).to(device)
            all_target_features = model.encode_image(curr_images)
    else:
        texts = tokenizer_funct(target_prompts).to(device)
        all_target_features = model.encode_text(texts)

    return all_target_features


def initialize_prompt(tokenizer, token_embedding, args, device):
    prompt_len = args["prompt_len"]

    # randomly optimize prompt embeddings
    prompt_ids = torch.randint(len(tokenizer.encoder), (args["prompt_bs"], prompt_len)).to(device)
    prompt_embeds = token_embedding(prompt_ids).detach()
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(" ".join(["<start_of_text>"] * prompt_len))
    dummy_ids = tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * args["prompt_bs"]).to(device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = copy.deepcopy(dummy_ids)
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = token_embedding(tmp_dummy_ids).detach()
    dummy_embeds.requires_grad = False
    
    return prompt_embeds, dummy_embeds, dummy_ids

def optimize_prompt_loop_ph2p(model, tokenizer, token_embedding, all_target_features, args, device):
    opt_iters = args["iter"]
    ta = 500  # Starting timestep, as per the algorithm description
    T = 1000  # Assuming the total number of timesteps
    lr = args["lr"]
    print_step = args["print_step"]
    batch_size = args["batch_size"]
    print_new_best = getattr(args, 'print_new_best', False)

    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device)
    best_sim = -1000 * args["loss_weight"]
    best_text = ""

    for step in range(opt_iters):
        # Random timestep selection between ta and T
        t = np.random.randint(ta, T)

        # Optimize using L-BFGS
        projected_embeds, _ = nn_project(prompt_embeds, token_embedding)

        def loss_fn(prompt_embeds_flat):
            prompt_embeds_reshaped = torch.tensor(prompt_embeds_flat).view_as(projected_embeds).to(device)
            prompt_embeds_reshaped.requires_grad = True

            # Perform the same operations as in your existing optimize loop
            # (this is a placeholder, you'll need to adapt your existing code to fit here)

            loss = compute_loss(model, prompt_embeds_reshaped, dummy_embeds, dummy_ids, all_target_features, t, args)
            grad = torch.autograd.grad(loss, prompt_embeds_reshaped)[0]
            return loss.item(), grad.flatten().cpu().numpy()

        # Using scipy's L-BFGS optimizer
        x0 = projected_embeds.detach().cpu().numpy().flatten()
        x, _, _ = fmin_l_bfgs_b(loss_fn, x0, maxfun=20, pgtol=1e-6)

        # Update prompt embeddings
        prompt_embeds = torch.tensor(x).view_as(projected_embeds).to(device)
        prompt_embeds.requires_grad = True

        # Check for new best cosine similarity and update best_text if found
        # (similar to existing optimize loop)

        if print_step is not None and (step % print_step == 0 or step == opt_iters-1):
            # Print progress message
            # (similar to existing optimize loop)

    if print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")

    return best_text

# Note: The function compute_loss needs to be implemented. It should calculate the loss as per the given equation
# and return the loss and gradient for L-BFGS optimization.

# The optimize_prompt function should now call optimize_prompt_loop_ph2p instead of optimize_prompt_loop_pez.

# 
# def optimize_prompt_loop_p2hp(model, tokenizer, token_embedding, all_target_features, args, device):
#     opt_iters = args["iter"]
#     lr = args["lr"]
#     print_step = args["print_step"]
#     print_new_best = args.get("print_new_best", False)
#     ta = 500  # Starting timestep for optimization
#     T = 1000  # Maximum timestep
# 
#     # Initialize prompt
#     prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device)
# 
#     best_sim = -np.inf
#     best_text = ""
# 
#     for step in range(opt_iters):
#         # Select diffusion timestep
#         t = random.randint(ta, T)
# 
#         # Forward projection
#         projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding)
# 
#         # Optimization using L-BFGS
#         def loss_func(embeds, x0):
#             embeds = embeds.reshape(x0.shape)
#             padded_embeds = dummy_embeds.detach().clone()
#             padded_embeds[dummy_ids == -1] = x0.reshape(-1, prompt_embeds.shape[-1])
#             logits_per_image, _ = model.forward_text_embedding(model, padded_embeds, dummy_ids, all_target_features)
#             loss = 1 - logits_per_image.mean(dim=0).max()
#             return loss.item(), torch.autograd.grad(loss, [x0])[0].cpu().detach().numpy()
# 
#         x0 = projected_embeds.detach()
#         x0.requires_grad = True
#         #x0 = projected_embeds.detach().cpu().numpy()
#         x, f, d = fmin_l_bfgs_b(func=loss_func, x0=x0.cpu().detach(), args=(x0), maxiter=1, iprint=-1)
# 
#         # Update prompt embeddings
#         prompt_embeds.data = torch.from_numpy(x).to(device)
# 
#         # Evaluate current prompt
#         decoded_text = decode_ids(nn_indices, tokenizer)[0]
#         with torch.no_grad():
#             padded_embeds = dummy_embeds.detach().clone()
#             padded_embeds[dummy_ids == -1] = prompt_embeds.reshape(-1, prompt_embeds.shape[-1])
#             logits_per_image, _ = model.forward_text_embedding(model, padded_embeds, dummy_ids, all_target_features)
#             cosim_score = logits_per_image.mean(dim=0).max().item()
# 
#         # Check for new best
#         if cosim_score > best_sim:
#             best_sim = cosim_score
#             best_text = decoded_text
#             if print_step is not None and print_new_best:
#                 print(f"New best cosine similarity: {best_sim}, Prompt: {best_text}")
# 
#         # Regular progress update
#         if print_step is not None and (step % print_step == 0 or step == opt_iters - 1):
#             print(f"Step: {step}, Cosine similarity: {cosim_score}, Prompt: {decoded_text}")
# 
#     # Final result
#     print(f"Best cosine similarity: {best_sim}, Best prompt: {best_text}")
#     return best_text


def optimize_prompt_loop_pez(model, tokenizer, token_embedding, all_target_features, args, device):
    opt_iters = args["iter"]
    lr = args["lr"]
    weight_decay = args["weight_decay"]
    print_step = args["print_step"]
    batch_size = args["batch_size"]
    print_new_best = getattr(args, 'print_new_best', False)

    # initialize prompt
    prompt_embeds, dummy_embeds, dummy_ids = initialize_prompt(tokenizer, token_embedding, args, device)
    p_bs, p_len, p_dim = prompt_embeds.shape

    # get optimizer
    input_optimizer = torch.optim.AdamW([prompt_embeds], lr=lr, weight_decay=weight_decay)

    best_sim = -1000 * args["loss_weight"]
    best_text = ""

    for step in range(opt_iters):
        # randomly sample sample images and get features
        if batch_size is None:
            target_features = all_target_features
        else:
            curr_indx = torch.randperm(len(all_target_features))
            target_features = all_target_features[curr_indx][0:batch_size]
            
        universal_target_features = all_target_features
        
        # forward projection
        projected_embeds, nn_indices = nn_project(prompt_embeds, token_embedding, print_hits=False)
        model

        # get cosine similarity score with all target features
        with torch.no_grad():
            # padded_embeds = copy.deepcopy(dummy_embeds)
            padded_embeds = dummy_embeds.detach().clone()
            padded_embeds[dummy_ids == -1] = projected_embeds.reshape(-1, p_dim)
            logits_per_image, _ = model.forward_text_embedding(model, padded_embeds, dummy_ids, universal_target_features)
            scores_per_prompt = logits_per_image.mean(dim=0)
            universal_cosim_score = scores_per_prompt.max().item()
            best_indx = scores_per_prompt.argmax().item()
        
        # tmp_embeds = copy.deepcopy(prompt_embeds)
        tmp_embeds = prompt_embeds.detach().clone()
        tmp_embeds.data = projected_embeds.data
        tmp_embeds.requires_grad = True
        
        # padding
        # padded_embeds = copy.deepcopy(dummy_embeds)
        padded_embeds = dummy_embeds.detach().clone()
        padded_embeds[dummy_ids == -1] = tmp_embeds.reshape(-1, p_dim)
        
        logits_per_image, _ = model.forward_text_embedding(model, padded_embeds, dummy_ids, target_features)
        cosim_scores = logits_per_image
        loss = 1 - cosim_scores.mean()
        loss = loss * args["loss_weight"]
        
        prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
        
        input_optimizer.step()
        input_optimizer.zero_grad()

        curr_lr = input_optimizer.param_groups[0]["lr"]
        cosim_scores = cosim_scores.mean().item()

        decoded_text = decode_ids(nn_indices, tokenizer)[best_indx]
        if print_step is not None and (step % print_step == 0 or step == opt_iters-1):
            per_step_message = f"step: {step}, lr: {curr_lr}"
            if not print_new_best:
                per_step_message = f"\n{per_step_message}, cosim: {universal_cosim_score:.3f}, text: {decoded_text}"
            print(per_step_message)

        if best_sim * args["loss_weight"] < universal_cosim_score * args["loss_weight"]:
            best_sim = universal_cosim_score
            best_text = decoded_text
            if print_new_best:
                print(f"new best cosine sim: {best_sim}")
                print(f"new best prompt: {best_text}")


    if print_step is not None:
        print()
        print(f"best cosine sim: {best_sim}")
        print(f"best prompt: {best_text}")

    return best_text


def optimize_prompt(model=None, preprocess=None, args=None, device=None, target_images=None, target_prompts=None):
    global run_args
    clip_model = 'ViT-L/14'
    pretrained = 'openai'
    if model == None or preprocess == None:
        print("preprocess is None")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=clip_model, pretrained="openai", device=device)
        setattr(model, 'forward_text_embedding', forward_text_embedding)
        setattr(model, 'encode_text_embedding', encode_text_embedding)
        #model, preprocess = clip.load(clip_model, 
    if args is None:
        args = run_args
    token_embedding = model.token_embedding
    tokenizer = open_clip.tokenizer._tokenizer
    tokenizer_funct = open_clip.tokenizer.tokenize
    #tokenizer_funct = open_clip.get_tokenizer(clip_model)

    # get target features
    all_target_features = get_target_feature(model, preprocess, tokenizer_funct, device, target_images=target_images, target_prompts=target_prompts)

    # optimize prompt
    learned_prompt = optimize_prompt_loop_p2hp(model, tokenizer, token_embedding, all_target_features, args, device)

    return learned_prompt
    

def measure_similarity(orig_images, images, ref_model, ref_clip_preprocess, device):
    with torch.no_grad():
        ori_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in orig_images]
        ori_batch = torch.concatenate(ori_batch).to(device)

        gen_batch = [ref_clip_preprocess(i).unsqueeze(0) for i in images]
        gen_batch = torch.concatenate(gen_batch).to(device)
        
        ori_feat = ref_model.encode_image(ori_batch)
        gen_feat = ref_model.encode_image(gen_batch)
        
        ori_feat = ori_feat / ori_feat.norm(dim=1, keepdim=True)
        gen_feat = gen_feat / gen_feat.norm(dim=1, keepdim=True)
        
        return (ori_feat @ gen_feat.t()).mean().item()

def forward_text_embedding(self, embeddings, ids, image_features, avg_text=False, return_feature=False):
    text_features = self.encode_text_embedding(self, embeddings, ids, avg_text=avg_text)

    if return_feature:
        return text_features

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    # logit_scale = self.logit_scale.exp()
    logits_per_image = image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text

def encode_text_embedding(self, text_embedding, ids, avg_text=False):
    cast_dtype = self.transformer.get_cast_dtype()

    x = text_embedding + self.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x, attn_mask=self.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = self.ln_final(x)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    if avg_text:
        x = x[torch.arange(x.shape[0]), :ids.argmax(dim=-1)]
        x[:, 1:-1]
        x = x.mean(dim=1) @ self.text_projection
    else:
        x = x[torch.arange(x.shape[0]), ids.argmax(dim=-1)] @ self.text_projection

    return x