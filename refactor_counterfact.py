import os
from setup import setup
import pandas as pd
pd.set_option('display.max_colwidth', None)
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()
import numpy as np

def get_full_prompt(row):
    return row["prompt"].format(row["subject"])

def combine_prompts(row): 
    return list(set(row["paraphrase_prompts"] + row["generation_prompts"] + [row["full_prompt"]]))

def get_refactored_counterfact_df():
    refactored_counterfact_path = "refactored_counterfact.json"
    if not os.path.exists(refactored_counterfact_path):
        print(f"Generating counterfact_df", flush=True)
        setup_module = setup()
        counterfact_df = setup_module["counterfact_df"]

        ## flatten nested fields
        refactor_df = pd.concat([counterfact_df.requested_rewrite.apply(pd.Series), counterfact_df.drop('requested_rewrite', axis=1)], axis=1)
        refactor_df["target_new"] = refactor_df["target_new"].str["str"]
        refactor_df["target_true"] = refactor_df["target_true"].str["str"]

        ## add complete prompt
        refactor_df["full_prompt"] = refactor_df.apply(get_full_prompt, axis = 1)

        ## union prompts & remove duplicates 
        refactor_df["all_prompts"] = refactor_df.apply(combine_prompts, axis = 1)

        ## drop unnecessary columns
        refactor_df.drop(columns = ['case_id', 'pararel_idx', 'paraphrase_prompts', 'neighborhood_prompts', 'attribute_prompts', 'generation_prompts'], inplace=True)
        ## save json
        refactor_df.to_json(refactored_counterfact_path, orient='records')
        print(f"Saved counterfact_df to {refactored_counterfact_path}", flush=True)
    else:
        print(f"Loading counterfact_df from {refactored_counterfact_path}", flush=True)
    
    refactor_df = pd.read_json(refactored_counterfact_path)
    return refactor_df

def split_refactored_counterfact_df(refactored_counterfact_path):
    refactor_df = pd.read_json(refactored_counterfact_path)
    refactor_dfs = np.array_split(refactor_df, 20)
    os.makedirs("refactor_df", exist_ok=False)
    for i, df in enumerate(refactor_dfs):
        df.to_json(os.path.join("refactor_df_dummy", f"part_{i}.json")) # orient='records'