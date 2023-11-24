#!/usr/bin/python3
import argparse

# Scientific packages
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()

# Utilities
from utils import (
    make_inputs,
    decode_tokens,
    find_token_range,
    
)
from setup import setup
from methods import *
from cache_of_hidden_representations import build_subject_cache, build_hs_cache, prompt_relation_before_subject, prompt_subject, prompt_relation, template, full_prompt, attribute

import os

import random
from torch.nn import functional as F
import time
random.seed(int(time.time()))

setup_module = setup()
mt = setup_module["mt"]
knowns_df = setup_module["knowns_df"]
stopwords0_ = setup_module["stopwords0_"]
E = mt.model.get_input_embeddings().weight.detach()

# def attribute_extraction(df, subject_cache, prompt_func):
#     records = []
#     for row_i, row in tqdm(df.iterrows()):
#         prompt = prompt_func(row)
#         subject = row.subject
#         # TODO (can change this if use new json)
#         attribute_tok_str = f" {attribute(row)}"
#         attribute_tok = make_inputs(mt.tokenizer, [attribute_tok_str])["input_ids"][0].cpu().item()
        
#         inp = make_inputs(mt.tokenizer, [prompt])
#         input_tokens = decode_tokens(mt.tokenizer, inp["input_ids"][0])
#         e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
#         e_range = [x for x in range(e_range[0], e_range[1])]
#         source_index = len(input_tokens) - 1
        
#         # set hooks to get ATTN and MLP outputs
#         hooks = set_act_get_hooks(mt.model, source_index, mlp=True, attn_out=True)
#         output = mt.model(**inp)
#         # remove hooks
#         remove_hooks(hooks)
        
#         for layer in range(mt.num_layers):
#             attn_out = mt.model.activations_[f'attn_out_{layer}'][0]
#             proj_attn = attn_out.matmul(E.T).cpu().numpy()
#             ind_attn = np.argsort(-proj_attn, axis=-1)
#             attribute_tok_rank_attn = np.where(ind_attn == attribute_tok)[0][0]
            
#             subj_hs = subject_cache[(subject, layer)][0]
#             proj_hs = subj_hs.matmul(E.T).cpu().numpy()
#             ind_hs = np.argsort(-proj_hs, axis=-1)
#             attribute_tok_rank_hs = np.where(ind_hs == attribute_tok)[0][0]
#             attribute_tok_rank_diff = attribute_tok_rank_hs - attribute_tok_rank_attn
            
#             records.append({
#                 "prompt": prompt,
#                 "subject": subject,
#                 "relation_id": row.relation_id,
#                 "attribute": attribute(row),
#                 "attribute_tok": attribute_tok,
#                 "attribute_tok_str": attribute_tok_str,
#                 "layer": layer,
#                 "attribute_tok_rank_attn": attribute_tok_rank_attn,
#                 "attribute_tok_rank_hs": attribute_tok_rank_hs,
#                 "attribute_tok_rank_diff": attribute_tok_rank_diff,
#                 "attribute_in_top_1_attn": attribute_tok == ind_attn[0],
#                 "attribute_in_top_1_hs": attribute_tok == ind_hs[0],
#                 })
#     return pd.DataFrame.from_records(records)

def attribute_extraction(df, k=10):
    records = []
    for row_i, row in tqdm(df.iterrows()):
        prompt = row.prompt
        subject = row.subject
        attribute_tok_str = row.attribute
        attribute_tok = make_inputs(mt.tokenizer, [attribute_tok_str])["input_ids"][0].cpu().item()

        inp = make_inputs(mt.tokenizer, [prompt])
        input_tokens = decode_tokens(mt.tokenizer, inp["input_ids"][0])
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
        e_range = [x for x in range(e_range[0], e_range[1])]
        source_index = len(input_tokens) - 1
        
        # set hooks to get ATTN and MLP outputs
        hooks = set_act_get_hooks(mt.model, source_index, mlp=True, attn_out=True)
        output = mt.model(**inp)
        # remove hooks
        remove_hooks(hooks)
        
        # probs = torch.softmax(output["logits"][:, -1], dim=1)
        # _, attribute_tok = torch.max(probs, dim=1)
        # attribute_tok = attribute_tok.cpu().item()
        # [attribute_tok_str] = decode_tokens(mt.tokenizer, [attribute_tok])
        
        for layer in range(mt.num_layers):
            # ATTN
            attn_out = mt.model.activations_[f'attn_out_{layer}'][0]
            proj = attn_out.matmul(E.T).cpu().numpy()
            ind = np.argsort(-proj, axis=-1)
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            top_k_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
            records.append({
                    "prompt": prompt,
                    "subject": subject,
                    "target_true": row.target_true,
                    "attribute_tok": attribute_tok,
                    "attribute": attribute_tok_str,
                    "layer": layer,
                    "proj_vec": "MHSA",
                    "top_k_preds": top_k_preds,
                    "attribute_tok_rank": attribute_tok_rank,
                    "attribute_tok_score": attribute_tok_score,
                    "attribute_in_top_1": attribute_tok_rank == 0,
                })
            
            # MLP
            mlp_out = mt.model.activations_[f'm_out_{layer}']
            proj = mlp_out.matmul(E.T).cpu().numpy()
            ind = np.argsort(-proj, axis=-1)
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            top_k_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
            records.append({
                    "prompt": prompt,
                    "subject": subject,
                    "target_true": row.target_true,
                    "attribute_tok": attribute_tok,
                    "attribute": attribute_tok_str,
                    "layer": layer,
                    "proj_vec": "MLP",
                    "top_k_preds": top_k_preds,
                    "attribute_tok_rank": attribute_tok_rank,
                    "attribute_tok_score": attribute_tok_score,
                    "attribute_in_top_1": attribute_tok_rank == 0,
                })
    return pd.DataFrame.from_records(records)        

def extraction_event(df):
    extraction_event_df = df.drop("top_k_preds", axis=1).drop_duplicates().groupby(["prompt", "subject", "attribute"])['attribute_in_top_1'].any().reset_index(name='attribute_in_top_1_event')
    extraction_event_df["attribute_in_top_1_event"].mean()

if __name__ == "__main__":
    df = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_tracing_data/_mistakes_sample.json")
    # df = knowns_df
    df = df[df["subject"] != 'Vladimír Růžička']
    df = df[df["subject"] != 'Dražen Petrović']
    # subject_prompt_cache = build_subject_cache(mt, df)
    # subject_only_cache = build_subject_cache(mt, df, prompt_subject)
    subject_prompt_results = attribute_extraction(df)
    # subject_only_results = attribute_extraction(df, subject_only_cache, prompt_subject)
    subject_prompt_results.to_json("/home/gamir/DER-Roei/dhgottesman/DFR/mistakes_attribute_extraction_rate.json")
    print(subject_prompt_results)