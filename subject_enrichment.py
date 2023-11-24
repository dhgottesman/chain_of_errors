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
    prediction,
)
from setup import setup
from methods import *
from cache_of_hidden_representations import build_hs_cache, prompt_relation_before_subject, prompt_subject, template, full_prompt, attribute, prompt_relation_before_subject_templates, prompt_relation_before_subject_relation_only, prompt_relation_before_subject_extended

import os

import random
from torch.nn import functional as F
import time
random.seed(int(time.time()))
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")


os.environ["AWS_PROFILE"] = "daniela"
setup_module = setup()
mt = setup_module["mt"]
knowns_df = setup_module["knowns_df"]
stopwords0_ = setup_module["stopwords0_"]
E = mt.model.get_input_embeddings().weight.detach()


def _attribute_rank_in_representation(hs_cache, k, prompts, attribute):
    for layer in list(range(mt.num_layers+1)):
        def top_preds(prompt, position):
            hs = hs_cache[(prompt, layer)][0][position]
            projs = hs.matmul(E.T).cpu().numpy()
            ind = np.argsort(-projs)
            return [decode_tokens(mt.tokenizer, [i])[0] for i in ind]
        
        for desc, prompt, position in prompts:
            record = {"layer": layer}
            preds = top_preds(prompt, position)
            record[desc] = prompt
            record[f"{desc}_position"] = position
            record[f"top_{k}_{desc}"] = preds[:k]
            record[f"{desc}_attribute_rank"] = preds.index(attribute)
            yield record

def last_subject_representation(df, hs_cache, k, prompt_desc, prompt_func):
    records = []
    # Projection of token representations across all layers
    for row_i, row in tqdm(df.iterrows()):
        # Prompt Variations
        row_prompts = []
        prompts = prompt_func(row)
        prompts = prompts if type(prompts) == list else [prompts]
        for prompt in prompts:
            inp = make_inputs(mt.tokenizer, [prompt])
            # Position is last subject token
            e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], row.subject)                
            row_prompts.append((f"{prompt_desc}", prompt, e_range[1]-1))
        
        # We are checking rank of attribute in last token of the prompt variations
        for record in _attribute_rank_in_representation(hs_cache, k, row_prompts, row.attribute):
            record["subject"] = row.subject
            record["attribute"] = row.attribute
            record["relation_id"] = row.relation_id
            records.append(record)
    return pd.DataFrame.from_records(records)

def last_representation(df, hs_cache, k, prompt_desc, prompt_func):
    records = []
    # Projection of token representations across all layers
    for row_i, row in tqdm(df.iterrows()):
        # Prompt Variations
        row_prompts = []
        prompts = prompt_func(row)
        prompts = prompts if type(prompts) == list else [prompts]
        for prompt in prompts:
            inp = make_inputs(mt.tokenizer, [prompt])
            # Position is last token
            row_prompts.append((f"{prompt_desc}", prompt, len(inp["input_ids"][0])-1))
        
        # We are checking rank of attribute in last token of the prompt variations
        for record in _attribute_rank_in_representation(hs_cache, k, row_prompts, row.attribute):
            record["subject"] = row.subject
            record["attribute"] = row.attribute
            record["relation_id"] = row.relation_id
            records.append(record)
    return pd.DataFrame.from_records(records)

def last_relation_before_subject_relation_only_representation(df, hs_cache, k):
    records = []
    # Projection of token representations across all layers
    for row_i, row in tqdm(df.iterrows()):
        # Prompt Variations
        row_prompts = []
        prompt_desc = "relation_before_subject_relation_only"
        prompts = prompt_relation_before_subject_relation_only(row)
        prompts = prompts if type(prompts) == list else [prompts]
        for prompt in prompts:
            # Cut off prompt until subject placeholder
            inp = make_inputs(mt.tokenizer, [prompt])
            # Position is last token
            row_prompts.append((f"{prompt_desc}", prompt, len(inp["input_ids"][0])-1))
        
        # We are checking rank of attribute in last token of the prompt variations
        for record in _attribute_rank_in_representation(hs_cache, k, row_prompts, row.attribute):
            record["subject"] = row.subject
            record["attribute"] = row.attribute
            record["relation_id"] = row.relation_id
            records.append(record)
    return pd.DataFrame.from_records(records)

def retrieve_top_k_enwiki_paragraphs(client, index="wikipedia", k=100):
    def retrieve(subject):
        res = client.search(index=index, body={
                    "query": {
                        "bool": {
                            "should": [
                                {"match": {"text": subject}},
                                {"match": {"title": subject}}
                            ]
                        }
                    },
                    "size": k
                })

        # Return aggregate title, text
        paragraphs = []
        if res["hits"]["total"]["value"] > 100:
            for hit in res["hits"]["hits"]:
                title = hit["_source"]["title"]
                text = hit["_source"]["text"]
                if subject in text or subject in title:
                    paragraphs.append(f"{title} {text}")
        return paragraphs
    return retrieve

# Tokenize, remove duplicate tokens, stopwords, and subwords. 
def context_tokenized_dedup(paragraphs):
    words = []
    for paragraph in paragraphs:
        new_tokens = mt.tokenizer(paragraph)
        words = words + list(set(decode_tokens(mt.tokenizer, new_tokens['input_ids'])))
    return words

# Remove stopwords and words shorter than 3 characters
def remove_stopwords(words):
    return [
        word for word in words 
        if word.strip() not in stopwords0_ and len(word.strip()) > 2
    ]

def extract_entity_in_continuation(row):
  continuation = row["next_25_tokens"][len(row["relation_before_subject"]):]
  doc = nlp(continuation)
  if len(doc.ents) == 0:
    return None
  return " ".join(remove_stopwords(doc.ents[0].text.split()))

def _attributes_rate(top_k_preds, context_tokenized_clean):
    top_k_preds_set = set([token.strip() for token in top_k_preds])
    context_tokenized_clean = set([token.strip() for token in context_tokenized_clean])
    predicted_attributes = top_k_preds_set.intersection(context_tokenized_clean)
    return len(predicted_attributes) / len(top_k_preds_set)

def attributes_tokens(client, df, index, k, prefix):
    attributes_df = df[["subject"]].drop_duplicates(["subject"])
    attributes_df[f"top_{k}_paragraphs"] = attributes_df["subject"].progress_apply(retrieve_top_k_enwiki_paragraphs(client, index, k))
    attributes_df.to_json(f"_top_{k}_paragraphs_{prefix}.json")
    print(f"Retrieved top {k} paragraphs", flush=True)

    attributes_df["context_tokenized_dedup"] = attributes_df[f"top_{k}_paragraphs"].progress_apply(context_tokenized_dedup)
    attributes_df = attributes_df.drop(columns=[f"top_{k}_paragraphs"])
    print(f"Context tokens deduped", flush=True)

    attributes_df["context_tokenized_clean"] = attributes_df["context_tokenized_dedup"].progress_apply(remove_stopwords)
    attributes_df = attributes_df.drop(columns=[f"context_tokenized_dedup"])
    print(f"Cleaned context tokens", flush=True)
    attributes_df.to_json(f"_attributes_tokens_{prefix}.json")
    return attributes_df

# attributes_df = attributes_tokens(client, last_subject_representation_df, index, k, prefix)
# attributes_df = pd.read_json(cache_path)
def subject_attributes_rate(rank_df, attributes_df, k):
    results_df = pd.merge(rank_df, attributes_df, on="subject")
    results_df["attributes_rate"] = results_df.progress_apply(lambda row: _attributes_rate(row[f"top_{k}_subject_only"], row["context_tokenized_clean"]), axis=1)
    print(f"Merged attributes rate", flush=True)
    return results_df

def random_token():
    vocab_tokens = list(mt.tokenizer.get_vocab().keys())
    return str(random.choice(vocab_tokens))

# We compare the first two prompts, the third prompt is the target we are ranking
def next_token_similarity_analysis(df):
    records = []
    for row_i, row in tqdm(df.iterrows()):
        base_prompt = template(row)
        rand1, rand2 = random_token(), random_token()
        # base_prompt, random_prompt, full_prompt
        prompts = [base_prompt, base_prompt.format(rand1), full_prompt(row), rand1, rand2]
        logits = [next_token(mt, prompt)[0] for prompt in prompts]
        cosim = [F.cosine_similarity(logits[0], l, axis=0).item() for l in logits[1:-2]]
        cosim_rand = F.cosine_similarity(logits[-1], logits[-2], axis=0).item()
        record = {
            "subject": row["subject"],
            "attribute": attribute(row), 
            "relation_id": row["relation_id"],
            "template": base_prompt,
            "random_prompt": prompts[1],
            "full_prompt": prompts[2],
            "cosim_base_random": cosim[0],
            "cosim_base_full_prompt": cosim[1],
            "cosim_random_random": cosim_rand,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)

def load_df_from_dir(directory):
    files = [f for f in os.listdir(directory) if f[-5:] == ".json"]
    df = None
    for f in files:
        fp = os.path.join(directory, f)
        if df is None:
            df = pd.read_json(fp)
        else:
            df = pd.concat([df, pd.read_json(fp)])
    return df.reset_index()

def get_next_tokens(prompt, num_of_tokens):
    inp = make_inputs(mt.tokenizer, [prompt])
    inp["max_new_tokens"] = num_of_tokens
    res = mt.model.generate(**inp)
    decoded_tokens = mt.tokenizer.batch_decode(res)
    return decoded_tokens[0]

def pred_attribute_rank(prompt_desc):
    def _pred_attribute_rank(row):
        inp = make_inputs(mt.tokenizer, [row[prompt_desc]])
        out = mt.model(**inp)["logits"]
        probs = torch.softmax(out[:, -1], dim=1)
        token_preds = np.argsort(-probs.cpu().numpy())
        token_str_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in token_preds[0]]
        return token_str_preds.index(row["attribute"])
    return _pred_attribute_rank

def contains_substring(k):
    return lambda row: row['attribute'].strip() in row[f"next_{k}_tokens"]

def attribute_index_in_continuation(k):
    def _attribute_index_in_continuation(row):
        next_token_ids = make_inputs(mt.tokenizer, [row[f"next_{k}_tokens"]])["input_ids"][0]
        token_str_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in next_token_ids]
        try:
            attribute_index = token_str_preds.index(row["attribute"])
        except Exception as e:
            attribute_index = -1
        prompt_token_ids = make_inputs(mt.tokenizer, [row["relation_before_subject"]])["input_ids"][0].cpu().numpy() 
        return -1 if attribute_index == -1 else (attribute_index - len(prompt_token_ids))
    return _attribute_index_in_continuation

def attribute_extraction(df, prompt_desc, prompt_func, k, attribute_func = lambda row: row.attribute):
    records = []
    for row_i, row in tqdm(df.iterrows()):
        prompt = prompt_func(row)
        subject = row.subject
        attribute_tok_str = attribute_func(row)
        if attribute_tok_str is None or attribute_tok_str == "":
            continue
        attribute_tok = make_inputs(mt.tokenizer, [attribute_tok_str])["input_ids"][0]
        if attribute_tok.size()[0] > 1:
            attribute_tok = attribute_tok[0].cpu().item()
        else:
            attribute_tok = attribute_tok.cpu().item()

        inp = make_inputs(mt.tokenizer, [prompt])
        input_tokens = decode_tokens(mt.tokenizer, inp["input_ids"][0])
        source_index = len(input_tokens) - 1
        
        # set hooks to get ATTN and MLP outputs
        hooks = set_act_get_hooks(mt.model, source_index, mlp=True, attn_out=True)
        output = mt.model(**inp)
        # remove hooks
        remove_hooks(hooks)
        
        for layer in range(mt.num_layers):
            # ATTN
            attn_out = mt.model.activations_[f'attn_out_{layer}'][0]
            proj = attn_out.matmul(E.T).cpu().numpy()
            ind = np.argsort(-proj, axis=-1)
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            top_k_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
            records.append({
                    prompt_desc: prompt,
                    "subject": subject,
                    "attribute_tok": attribute_tok,
                    "attribute": attribute_tok_str,
                    "layer": layer,
                    "proj_vec": "MHSA",
                    f"top_{k}_preds": top_k_preds,
                    "attribute_tok_rank": attribute_tok_rank,
                    "attribute_tok_score": attribute_tok_score,
                    "attribute_in_top_1": attribute_tok_rank == 0,
                    "relation_id": row.relation_id
                })
            
            # MLP
            mlp_out = mt.model.activations_[f'm_out_{layer}']
            proj = mlp_out.matmul(E.T).cpu().numpy()
            ind = np.argsort(-proj, axis=-1)
            attribute_tok_rank = np.where(ind == attribute_tok)[0][0]
            attribute_tok_score = proj[ind[attribute_tok_rank]]
            top_k_preds = [decode_tokens(mt.tokenizer, [i])[0] for i in ind[:k]]
            records.append({
                    prompt_desc: prompt,
                    "subject": subject,
                    "attribute_tok": attribute_tok,
                    "attribute": attribute_tok_str,
                    "layer": layer,
                    "proj_vec": "MLP",
                    f"top_{k}_preds": top_k_preds,
                    "attribute_tok_rank": attribute_tok_rank,
                    "attribute_tok_score": attribute_tok_score,
                    "attribute_in_top_1": attribute_tok_rank == 0,
                    "relation_id": row.relation_id
                })
    return pd.DataFrame.from_records(records) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Add paragraphs from a JSON file to an Elasticsearch index.')
    parser.add_argument('--mistakes', help='Path to mistakes examples')
    parser.add_argument('--mistakes_ar_cache', help='Path to mistakes attributes rate cache')
    parser.add_argument('--knowns_ar_cache', help='Path to knowns attributes rate cache')
    parser.add_argument('--k', type=int, help='Number of paragraphs to retrieve')
    parser.add_argument('--layer', default=-1, type=int, help='Restrict analyses to specific layer')
    parser.add_argument('--plot-mean-ar', action="store_true")
    parser.add_argument('--plot_ar_histogram', action="store_true")
    parser.add_argument('--host', help='Elastic Search hostname')
    parser.add_argument('--port', default=9200, help='Port number')
    parser.add_argument('--region', default=None, help='The region for AWS ES Host. '
                        'If set, we use AWS_PROFILE credentials to connect to the host')
    parser.add_argument('--index', help='Name of index to create')
    parser.add_argument('--idx', type=int, help='idx in small df')

    args = parser.parse_args()  

    knowns_labeled = pd.read_csv("/home/gamir/DER-Roei/dhgottesman/DFR/knowns_labeled.csv")
    knowns_labeled = knowns_labeled[["relation_before_subject", "attributes_rate", "labels"]]
    knowns_examples = pd.read_csv("/home/gamir/DER-Roei/dhgottesman/DFR/knowns_examples.csv")

    mistakes_labeled = pd.read_csv("/home/gamir/DER-Roei/dhgottesman/DFR/mistakes_labeled.csv")
    mistakes_labeled = mistakes_labeled[["relation_before_subject", "attributes_rate", "label"]]
    mistakes_examples = pd.read_csv("/home/gamir/DER-Roei/dhgottesman/DFR/mistake_examples_continuation_entity_extraction.csv")
    mistakes_examples["relation_before_subject_attribute_rank"] = mistakes_examples["attribute_tok_rank"]
    mistakes_extraction_event = mistakes_examples.groupby(["relation_before_subject"])['attribute_in_top_1'].any().reset_index(name='attribute_in_top_1_event')
    mistakes_examples = mistakes_examples.groupby('relation_before_subject').agg({'relation_before_subject_attribute_rank': 'min'})
    mistakes_examples = pd.merge(mistakes_examples, mistakes_extraction_event, on=["relation_before_subject"])

    knowns_labeled["label"] = knowns_labeled["labels"]
    knowns_labeled = knowns_labeled.drop("labels", axis=1)
    knowns_labeled = knowns_labeled.dropna(subset=["label"])
    mistakes_labeled = mistakes_labeled.dropna(subset=["label"])

    knowns_labeled["label_binary"] = 1
    mistakes_labeled["label_binary"] = 0

    knowns_examples = knowns_examples[["relation_before_subject", "attribute_in_top_1_event", "relation_before_subject_attribute_rank"]]
    mistakes_examples = mistakes_examples[["relation_before_subject", "attribute_in_top_1_event", "relation_before_subject_attribute_rank"]]

    knowns_labeled = pd.merge(knowns_labeled, knowns_examples, on=["relation_before_subject"])
    mistakes_labeled = pd.merge(mistakes_labeled, mistakes_examples, on=["relation_before_subject"])

    knowns_features = knowns_labeled[['attributes_rate', 'attribute_in_top_1_event','relation_before_subject_attribute_rank', 'label_binary']]
    mistakes_features = mistakes_labeled[['attributes_rate', 'attribute_in_top_1_event','relation_before_subject_attribute_rank', 'label_binary']]

    data = pd.concat([knowns_features, mistakes_features])
    data = data.sample(frac=1).reset_index(drop=True)

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    from sklearn import tree
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    # Load dataset
    X = data.drop("label_binary", axis=1)
    y = data["label_binary"]

    # Split the data into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a decision tree regressor
    clf = tree.DecisionTreeClassifier(max_depth=3)

    # Train the regressor on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the regressor
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error: ", mse)

    plt.figure(figsize=(30, 15))
    tree.plot_tree(clf, feature_names=list(X.columns), filled=True, fontsize=12)
    plt.savefig('foo.png')

    from sklearn.metrics import accuracy_score
    res_pred = clf.predict(X_test)
    score = accuracy_score(y_test, res_pred)


    # mistakes_examples = pd.read_csv("/home/gamir/DER-Roei/dhgottesman/DFR/mistakes_examples.csv")
    # mistakes_examples["continuation_entity"] = mistakes_examples.apply(extract_entity_in_continuation, axis=1)
    # df = attribute_extraction(mistakes_examples, "relation_before_subject", lambda row: row.relation_before_subject, 100, lambda row: row.continuation_entity)
    # df.to_csv("mistake_examples_continuation_entity_extraction.csv")


    # mistakes_rank_rate = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/combined_rank_rate_data/mistakes_rank_rate.json")
    # mistakes_extraction = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/extraction_data/mistakes_attribute_extraction.json")
    # mistakes_extraction_event = mistakes_extraction.groupby(["subject", "relation_id", "attribute"])['attribute_in_top_1'].any().reset_index(name='attribute_in_top_1_event')
    # mistakes_combined = pd.merge(mistakes_rank_rate, mistakes_extraction_event, on=["subject", "relation_id", "attribute"])
    # mistakes_prompts = mistakes_combined.loc[mistakes_combined.groupby(["subject", "attribute", "relation_id"])['relation_before_subject_pred_attribute_rank'].idxmin()][["relation_before_subject"]]
    # mistakes_combined = pd.merge(mistakes_combined, mistakes_prompts, on="relation_before_subject")

    # k = 25

    # df = mistakes_combined[["relation_before_subject", "attribute"]].drop_duplicates()
    # smaller_dfs = np.array_split(df, 10)  
    # unique_prompts = smaller_dfs[args.idx]

    # unique_prompts[f"next_{k}_tokens"] = unique_prompts["relation_before_subject"].progress_apply(lambda row: get_next_tokens(row, k))
    # unique_prompts[f"attribute_in_next_{k}_tokens"] = unique_prompts.apply(contains_substring(k), axis=1)
    # unique_prompts.to_json(f"/home/gamir/DER-Roei/dhgottesman/DFR/important/flow_analysis/mistakes_examples/part_{args.idx}.json")

    """
    # Combine ranks_rate data with extraction data to plot
    knowns_rank_rate = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/combined_rank_rate_data/knowns_rank_rate.json")
    knowns_extraction = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/extraction_data/knowns_attribute_extraction.json")
    # This was computed using knowns_relation_before_subject_attribute_leq_3_idx.json
    knowns_pred_rank = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/relation_before_subject_and_extended_pred_attribute_rank/knowns_relation_before_subject_and_extended_pred_attribute_rank.json")
    knowns_extraction_event = knowns_extraction.groupby(["subject", "relation_id", "attribute"])['attribute_in_top_1'].any().reset_index(name='attribute_in_top_1_event')
    print("knowns % extraction_event", knowns_extraction_event["attribute_in_top_1_event"].astype(int).mean())
    knowns_combined = pd.merge(knowns_rank_rate, knowns_extraction_event, on=["subject", "relation_id", "attribute"])
    knowns_combined = pd.merge(knowns_combined, knowns_pred_rank, on=["subject", "relation_id", "attribute", "relation_before_subject"])

    mistakes_rank_rate = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/combined_rank_rate_data/mistakes_rank_rate.json")
    mistakes_extraction = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/extraction_data/mistakes_attribute_extraction.json")
    mistakes_extraction_event = mistakes_extraction.groupby(["subject", "relation_id", "attribute"])['attribute_in_top_1'].any().reset_index(name='attribute_in_top_1_event')
    mistakes_combined = pd.merge(mistakes_rank_rate, mistakes_extraction_event, on=["subject", "relation_id", "attribute"])
    mistakes_prompts = mistakes_combined.loc[mistakes_combined.groupby(["subject", "attribute", "relation_id"])['relation_before_subject_pred_attribute_rank'].idxmin()][["relation_before_subject"]]
    mistakes_combined = pd.merge(mistakes_combined, mistakes_prompts, on="relation_before_subject")
    print("mistakes % extraction_event", mistakes_extraction_event["attribute_in_top_1_event"].astype(int).mean())

    tmp = knowns_combined.groupby('relation_before_subject').agg({'subject_only_attribute_rank': 'min', 'relation_before_subject_attribute_rank': 'min', 'attributes_rate': 'max'})
    tmp = mistakes_combined.groupby('relation_before_subject').agg({'subject_only_attribute_rank': 'min', 'relation_before_subject_attribute_rank': 'min', 'attributes_rate': 'max'})
    tmp["subject_only_attribute_rank"].corr(tmp["attributes_rate"])
    tmp["relation_before_subject_attribute_rank"].corr(tmp["attributes_rate"])
    tmp["attributes_rate"].mean()

    """
    
    """
    # Compute correlations
    """


    # # Compute extraction event for knowns relation_before_subject_extended
    # df = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_before_subject_and_extended_pred_attribute_rank/knowns_relation_before_subject_and_extended_pred_attribute_rank.json")
    # extraction_df = attribute_extraction(df, "relation_before_subject_extended", lambda row: row.relation_before_subject_extended, 100)
    # extraction_df.to_json("extraction_data/knowns_extended_attribute_extraction.json")

    # TODO: note that i accidentally ran knowns on good prompts and not on the leq_3, should use knowns extended extraction so
    # it will filter to leq_3.
 
    # TODO: note that i accidentally ran knowns on good prompts and not on the leq_3
    """
    # Compute extraction events

    df = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/combined_rank_rate_data/mistakes_rank_rate.json")
    df = df[["subject", "attribute", "relation_id", "relation_before_subject"]].drop_duplicates()
    extraction_df = attribute_extraction(df, "relation_before_subject", lambda row: row.relation_before_subject, 100)
    extraction_df.to_json("extraction_data/mistakes_attribute_extraction.json")
    """

    # TODO: note that i accidentally ran knowns on good prompts and not on the leq_3
    """
    # Compute attribute rate data
    
    k = 100

    mistakes_ranks = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/combined_rank_data/mistakes_attribute_ranks.json")
    knowns_ranks = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/combined_rank_data/knowns_attribute_ranks.json")

    mistakes_attributes = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_tracing_data/_attributes_tokens_mistakes.json")
    knowns_attributes = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_tracing_data/_attributes_tokens_knowns.json")

    mistakes_rank_rate = subject_attributes_rate(mistakes_ranks, mistakes_attributes, k)
    mistakes_rank_rate.to_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/combined_rank_rate_data/mistakes_rank_rate.json")
    
    knowns_rank_rate = subject_attributes_rate(knowns_ranks, knowns_attributes, k)
    knowns_rank_rate.to_json("/home/gamir/DER-Roei/dhgottesman/DFR/important/combined_rank_rate_data/knowns_rank_rate.json")
    """

    # TODO: note that i accidentally ran knowns on good prompts and not on the leq_3
    # note this doesn't include extended rank
    """
    # Combine all the knowns attribute rank data 

    knowns_relation_before_subject = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/good_prompts/knowns_relation_before_subject_good_prompts")
    knowns_relation_before_subject = knowns_relation_before_subject.drop(["level_0", "index"], axis=1)
    knowns_relation_before_subject_relation_only = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/knowns_relation_before_subject_relation_only")
    knowns_relation_before_subject_relation_only = knowns_relation_before_subject_relation_only.drop(["index"], axis=1)
    knowns_subject_only = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/knowns_subject_only.json")

    knowns_relation_before_subject["relation_before_subject_relation_only"] = knowns_relation_before_subject.progress_apply(lambda row: row["relation_before_subject"][:row["relation_before_subject"].find(row["subject"])-1], axis=1)
    combined = pd.merge(knowns_relation_before_subject_relation_only, knowns_subject_only, on=["subject", "attribute", "relation_id", "layer"])
    combined = pd.merge(combined, knowns_relation_before_subject, on=["layer", "subject", "attribute", "relation_id", "relation_before_subject_relation_only"])
    
    combined.to_json("combined_rank_data/knowns_attribute_ranks.json")
    """

    """
    # Combine all the mistake attribute rank data 
    mistakes_relation_before_subject = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/mistakes_no_good_relation_before_subject_pred_attribute_rank.json")
    mistakes_relation_before_subject = mistakes_relation_before_subject.drop("index", axis=1)
    mistakes_relation_before_subject_relation_only = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/mistakes_relation_before_subject_relation_only")
    mistakes_relation_before_subject_relation_only = mistakes_relation_before_subject_relation_only.drop("index", axis=1)
    mistakes_subject_only = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/mistakes_subject_only.json")

    mistakes_relation_before_subject["relation_before_subject_relation_only"] = mistakes_relation_before_subject.progress_apply(lambda row: row["relation_before_subject"][:row["relation_before_subject"].find(row["subject"])-1], axis=1)
    mistakes_relation_before_subject = mistakes_relation_before_subject.drop("layer", axis=1)
    combined = pd.merge(mistakes_relation_before_subject_relation_only, mistakes_subject_only, on=["subject", "attribute", "relation_id", "layer"])
    combined = pd.merge(combined, mistakes_relation_before_subject, on=["subject", "attribute", "relation_id", "relation_before_subject_relation_only"])

    combined.to_json("combined_rank_data/mistakes_attribute_ranks.json")
    """

    """
    # Filter out good examples from mistakes and compute attribute rank in next token
    good = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_before_subject_and_extended_pred_attribute_rank/mistakes_relation_before_subject_and_extended_pred_attribute_rank.json")
    mistakes = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/mistakes_relation_before_subject")
    mistakes_filtered = pd.merge(mistakes, good[["subject", "attribute", "relation_id"]], on=["subject", "attribute", "relation_id"], how='outer', indicator=True)
    mistakes_filtered = mistakes_filtered[mistakes_filtered['_merge'] == 'left_only']
    mistakes_filtered = mistakes_filtered.drop("_merge", axis=1)

    unique_prompts = mistakes_filtered.drop_duplicates("relation_before_subject")
    unique_prompts["relation_before_subject_pred_attribute_rank"] = unique_prompts.progress_apply(pred_attribute_rank("relation_before_subject"), axis=1)
    unique_prompts.to_json(f"/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/mistakes_no_good_relation_before_subject_pred_attribute_rank.json")
    """

    """
    # Attribute rank in next token for relation_before_subject and relation_before_subject_extended
    
    df = pd.read_json("knowns_relation_before_subject_attribute_leq_3_idx.json")
    df = pd.read_json("mistakes_relation_before_subject_attribute_leq_3_idx.json")

    df["relation_before_subject_pred_attribute_rank"] = df.progress_apply(pred_attribute_rank("relation_before_subject"), axis=1)
    df["relation_before_subject_extended_pred_attribute_rank"] = df.progress_apply(pred_attribute_rank("relation_before_subject_extended"), axis=1)
    df.to_json(f"/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/knowns_relation_before_subject_and_extended_pred_attribute_rank.json")
    """
    
    """
    # Keep relation_before_subject that lead to continuations containing the attribute within 3 tokens we use this so we can refine the relation in order to compute attribute extraction metrics
    
    df = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/knowns_relation_before_subject_good_prompts")
    df = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/mistakes_relation_before_subject_good_prompts")

    df["continuation_attribute_index"] = df.progress_apply(attribute_index_in_continuation(25), axis=1)
    tmp = df[(df["continuation_attribute_index"] > -1) & (df["continuation_attribute_index"] <= 3)]
    tmp2 = tmp.loc[tmp.groupby(["subject", "attribute", "relation_id"])['continuation_attribute_index'].idxmin()]
    tmp2["relation_before_subject_extended"] = tmp2.progress_apply(lambda row: row["next_25_tokens"][:row["next_25_tokens"].index(row["attribute"])], axis=1)
    tmp2[["subject", "attribute", "relation_id", "relation_before_subject", "relation_before_subject_extended"]].to_json("knowns_relation_before_subject_attribute_leq_3_idx.json")
    tmp2.to_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_before_subject_and_extended_pred_attribute_rank/knowns_relation_before_subject_attribute_leq_3_idx.json")
    """

    """
    # Keep relation_before_subject prompts that lead to continuations containing the attribute
    
    k = 25

    df = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/knowns_relation_before_subject")
    df = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/mistakes_relation_before_subject")

    unique_prompts = df.drop_duplicates("relation_before_subject")

    smaller_dfs = np.array_split(unique_prompts, 10)  
    unique_prompts = smaller_dfs[args.idx]

    unique_prompts[f"next_{k}_tokens"] = unique_prompts["relation_before_subject"].progress_apply(lambda row: get_next_tokens(row, k))
    unique_prompts[f"attribute_in_next_{k}_tokens"] = unique_prompts.apply(contains_substring(k), axis=1)
    good_prompts = unique_prompts[unique_prompts[f"attribute_in_next_{k}_tokens"]][["relation_before_subject", f"next_{k}_tokens"]]
    result = pd.merge(df, good_prompts, on=["relation_before_subject"])
    result.to_json(f"/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/knowns_relation_before_subject_good_prompts/part_{args.idx}.json")
    """

    """ 
    # Attribute ranks in last token or last subject token of prompt variations
    
    df = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_tracing_data/_formatted_known_1000.json")
    df = pd.read_json("/home/gamir/DER-Roei/dhgottesman/DFR/relation_tracing_data/_mistakes_sample.json")  
    
    prompt_desc, prompt_func = "relation_before_subject_relation_only", prompt_relation_before_subject_relation_only
    hs_cache = build_hs_cache(mt, df, prompt_func)
    smaller_dfs = np.array_split(df, 10)  
    result = last_representation(smaller_dfs[args.idx], hs_cache, 100, prompt_desc, prompt_func)
    result.to_json(f"/home/gamir/DER-Roei/dhgottesman/DFR/rank_dfs/knowns_relation_before_subject_relation_only/part_{args.idx}.json")
    """














    

