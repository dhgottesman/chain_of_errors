from methods import *
from enum import Enum
# Scienfitic packages
import pandas as pd
pd.set_option('display.max_colwidth', None)
import torch
from tqdm import tqdm
torch.set_grad_enabled(False)
tqdm.pandas()
import argparse
# Utilities
from utils import (
    make_inputs,
    decode_tokens,
    next_token_probs
)
import math
from setup import setup
from refactor_counterfact import get_refactored_counterfact_df, split_refactored_counterfact_df

## CONSTANTS
ENTROPY_THRESHOLD = 3
ENTROPY_RATIO_THRESHOLD = 0.75

class SplitType(Enum):
    NONE = 0
    ENTROPY = 1
    MIN_EXCLUDED_ENTROPY = 2
    MIN_EXCLUDED_ENTROPY_RATIO = 3
    RETRIVAL_TEST = 4 ##NotImplementedError

    def __str__(self):
        return self.name
    
    @staticmethod
    def from_string(s):
        try:
            return SplitType[s]
        except KeyError:
            raise ValueError()

class SampleType(Enum):
    SOFT = 0
    HARD = 1

def entropy(prob_vector):
    entropy_val = 0.0
    for prob in prob_vector:
        if prob > 0:
            entropy_val -= prob * math.log2(prob)
    return entropy_val

# load model and counterfact dataset
setup_module = setup()
mt = setup_module["mt"]
stopwords0_ = setup_module["stopwords0_"]


def combine_answers(row, k):
    answers = set()
    for prompt in row["all_prompts"]:
        inp = make_inputs(mt.tokenizer, [prompt])
        answer_t, base_score, _ = trace_with_proj(mt.model, inp, None, k)
        answer = decode_tokens(mt.tokenizer, answer_t)
        answers.update([y.strip().lower() for y in answer if y.strip().lower() not in stopwords0_ and len(y.strip()) > 2])
    return list(answers)

def true_target_not_in_all_answers(row, k):
    return row['target_true'].strip().lower() not in row[str.format("top_{}_answers_wo_order", k)]

def add_top_k_answers_wo_order(df, k):
    df[str.format("top_{}_answers_wo_order", k)] = df.apply(lambda r: combine_answers(r, k), axis=1)
    df[str.format("target_not_in_top_{}_answers_wo_order", k)] = df.apply(lambda r: true_target_not_in_all_answers(r, k), axis = 1)
    return df

def get_next_tokens(row, num_of_tokens):
    inp = make_inputs(mt.tokenizer, [row["full_prompt"]])
    inp["max_new_tokens"] = num_of_tokens
    res = mt.model.generate(**inp)
    decoded_tokens = mt.tokenizer.batch_decode(res)
    return decoded_tokens

def get_entropy(row):
    inp = make_inputs(mt.tokenizer, [row["full_prompt"]])
    probs = next_token_probs(mt.model, inp).cpu().data.numpy()
    return entropy(probs[0])

def enrich_dataset(refactored_df):
    # refactored_cf = get_refactored_counterfact_df()
    enriched_df = add_top_k_answers_wo_order(refactored_df, 50)
    print("Added add_top_50_answers_wo_order", flush=True)
    enriched_df["next_25_tokens"] = enriched_df.apply(lambda r: get_next_tokens(r, 25), axis=1)
    enriched_df["entropy"] = enriched_df.apply(get_entropy, axis=1)
    print("Added next_25_tokens", flush=True)
    return enriched_df

def exclude_tokens_from_prompt_entropy(row):
    #tokens = mt.tokenizer(tokens)
    words = row["full_prompt"].split(" ")
    excluded_words_entropy = dict()
    for i, word in enumerate(words):
        if (word in stopwords0_):
            continue
        new_prompt = " ".join(words[:i] + words[i+1:])
        inp = make_inputs(mt.tokenizer, [new_prompt])
        probs = next_token_probs(mt.model, inp).cpu().data.numpy()
        excluded_words_entropy[new_prompt] = entropy(probs[0])
    row["excluded_entropy"] = excluded_words_entropy
    row["min_excluded_entropy"] = min([excluded_words_entropy[key] for key in excluded_words_entropy])
    return row

def create_dataset(start, end, filter_out_correct_answers, enrich, split_type, enriched_dir, read_from_dir):
    if not enrich: 
        if split_type == SplitType.NONE:
            return
        else:
            enrich_prefix = "enriched_"
            files = [f for f in os.listdir(enriched_dir) if os.path.splitext(f)[1] == ".json" and "labled_" not in os.path.basename(f)]
            print(f"Files for smaples split: {files}", flush=True)
            for f in files[start: end]:
                print(f"Reading {os.path.join(enriched_dir, f)}", flush=True)
                df = pd.read_json(os.path.join(enriched_dir, f))
                if filter_out_correct_answers and "false_examples_only" not in os.path.basename(f):
                    df = df.loc[df.target_not_in_top_50_answers_wo_order] 
                    enrich_prefix+="false_examples_only_"
                df = df.apply(lambda r:label_samples(r, split_type, ENTROPY_THRESHOLD, ENTROPY_RATIO_THRESHOLD), axis=1) 
                enrich_prefix += f"labled_{split_type.name}_"
                file_path = os.path.join(enriched_dir, f"{enrich_prefix}{f}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                df.to_json(file_path, orient='records')    
    else:
        files = [f for f in os.listdir(read_from_dir) if os.path.splitext(f)[1] == ".json"]
        print(f"Files for enrichment: {files}", flush=True)
        files.sort()
        os.makedirs(enriched_dir, exist_ok=True)
        for f in files[start: end]:
            print(f"Reading {os.path.join(read_from_dir, f)}", flush=True)
            df = pd.read_json(os.path.join(read_from_dir, f))
            df = enrich_dataset(df)
            enrich_prefix = "enriched_"
            if filter_out_correct_answers:
                df = df.loc[df.target_not_in_top_50_answers_wo_order] 
                enrich_prefix+="false_examples_only_"
            df["entropy"] = df.apply(get_entropy, axis=1)
            df = df.apply(exclude_tokens_from_prompt_entropy, axis=1)
            if split_type != SplitType.NONE:
                df = df.apply(lambda r:label_samples(r, split_type, ENTROPY_THRESHOLD, ENTROPY_RATIO_THRESHOLD), axis=1) 
                enrich_prefix += f"labled_{split_type.name}_"
            file_path = os.path.join(enriched_dir, f"{enrich_prefix}{f}")
            if os.path.exists(file_path):
                os.remove(file_path)
            df.to_json(file_path, orient='records')

def label_samples(row, split_type, entropy_threshold, entropy_ratio_threshold): 
    if split_type == SplitType.ENTROPY:
        row["label"] = SampleType.HARD if row["entropy"] <= entropy_threshold else SampleType.SOFT # if entropy is small, the model is more confident, so hard mistake
        return row
    elif split_type == SplitType.MIN_EXCLUDED_ENTROPY:
        row["label"] = SampleType.HARD if row["entropy"] < row["min_excluded_entropy"] else SampleType.SOFT # if entropy is smaller than entropy without a word, the model is not confused, so hard mistake
        return row
    elif split_type == SplitType.MIN_EXCLUDED_ENTROPY_RATIO:
        row["label"] = SampleType.HARD if row["entropy"]/row["min_excluded_entropy"] < entropy_ratio_threshold else SampleType.SOFT
        return row
    else: #SplitType.RETRIVAL_TEST
        raise NotImplementedError("not implemented yet")

# first run refactor_counterfact to load counterfact db in correct format and then run this file 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()     
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--filter_out_correct_answers', type=bool)
    parser.add_argument('--split_type', type=lambda st: SplitType[st], choices=list(SplitType)) # Hard/Soft labeling logic. If 0 (SplitType.NONE), label column is not added to df
    parser.add_argument('--enrich', type=bool) # Full enrichment flow. 
    parser.add_argument('--enriched_dir', type=str)
    parser.add_argument('--read_from_dir', type=str)

    args = parser.parse_args()
    create_dataset(args.start, args.end, args.filter_out_correct_answers, args.enrich, args.split_type, args.enriched_dir, args.read_from_dir)