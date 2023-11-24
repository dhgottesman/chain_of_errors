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

def prompt_relation_before_subject(row):
    return [template.format(row.subject) for template in row.relation_before_subject_templates]

def prompt_relation_before_subject_templates(row):
    return row.relation_before_subject_templates

def prompt_relation_before_subject_relation_only(row):
    return [template[:template.find(" {}")] for template in row.relation_before_subject_templates]

def prompt_relation_before_subject_extended(row):
    return row.relation_before_subject_extended

def prompt_subject(row):
    return row.subject

def template(row):
    return row.template

def full_prompt(row):
    return row.prompt

def attribute(row):
    return row.attribute

# create a cache of hidden states
def build_hs_cache(mt, df, prompt_func=prompt_subject):
    hs_cache = {}
    layers_to_cache = list(range(mt.num_layers+1))
    for row_i, row in tqdm(df.iterrows()):
        def populate(prompt):
            inp = make_inputs(mt.tokenizer, [prompt])
            output = mt.model(**inp, output_hidden_states = True)

            for layer in layers_to_cache:
                if (prompt, layer) not in hs_cache:
                    hs_cache[(prompt, layer)] = []
                hs_cache[(prompt, layer)].append(output["hidden_states"][layer][0])
        # Need to try all variations of relation_before_subject prompts
        prompts = prompt_func(row)
        prompts = prompts if type(prompts) == list else [prompts]
        for prompt in prompts:
            populate(prompt)

    print(len(hs_cache))
    return hs_cache

# create a cache of subject representations
def build_subject_cache(mt, df, prompt_func=full_prompt):
    layers_to_cache = list(range(mt.num_layers))
    subject_cache = {}
    for row_i, row in tqdm(df.iterrows()):
        prompt = prompt_func(row)
        subject = row.subject
        
        inp = make_inputs(mt.tokenizer, [prompt])
        e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
        e_range = [x for x in range(e_range[0], e_range[1])]
        
        output = mt.model(**inp, output_hidden_states = True)
        
        probs = torch.softmax(output["logits"][:, -1], dim=1)
        base_score, answer_t = torch.max(probs, dim=1)
        base_score = base_score.cpu().item()
        [answer] = decode_tokens(mt.tokenizer, answer_t)
        
        for layer in layers_to_cache:
            if (subject, layer) not in subject_cache:
                subject_cache[(subject, layer)] = []
            subject_cache[(subject, layer)].append(output["hidden_states"][layer+1][0, e_range[-1]])

    print(len(subject_cache))
    return subject_cache

