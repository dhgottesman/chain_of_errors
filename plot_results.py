#!/usr/bin/python3
# Scientific packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def mistakes_vs_correct_attributes_rate_layer(mistakes, knowns):
    mistakes["label"] = "mistakes"
    knowns["label"] = "correct"
    # Create a box plot using Seaborn
    plt.figure(figsize=(30, 6))  # Adjust the figure size as needed
    sns.boxplot(x='layer', y='attributes_rate', hue='label', data=pd.concat([mistakes, knowns]))

    # Customize the plot
    plt.title('Correct vs Mistakes: Distribution of Attribute Rate by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Attributes Rate')
    plt.legend(title='Label', loc='upper right')
    plt.savefig('attributes_rate_mistakes_correct.png')

def attributes_rank(df, prefix):
    # Create a box plot using Seaborn
    plt.figure(figsize=(30, 16))  # Adjust the figure size as needed

    # Group the data by 'layer' and calculate the differences between the columns
    df['diff_rbs_subject'] = df['relation_before_subject_attribute_rank'] - df['subject_only_attribute_rank']
    df['diff_rbs_relation'] = df['relation_before_subject_attribute_rank'] - df['relation_only_attribute_rank']

    # Melt the DataFrame to plot multiple variables
    melted_df = pd.melt(df, id_vars=['layer', 'label'], value_vars=['relation_before_subject_attribute_rank', 'diff_rbs_subject', 'diff_rbs_relation'], var_name='Attribute', value_name='Value')

    # Create a box plot
    sns.boxplot(x='layer', y='Value', hue='Attribute', data=melted_df)

    # Customize the plot
    plt.title(f'{prefix}: Attribute Ranks by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Attribute Rank Difference')
    plt.yscale("log")
    plt.legend(title='Attribute', loc='upper right')
    plt.savefig(f'figures/attributes_rank_{prefix}.png')

def cosine_similarities(mistakes, knowns):
    # Concatenate 'mistakes' and 'correct' DataFrames
    mistakes["label"]= "mistakes"
    knowns["label"] = "correct"
    combined_df = pd.concat([mistakes, knowns])
    combined_df2 = combined_df.copy()
    combined_df2["label"] = "baseline"
    combined_df2["cosim_base_full_prompt"] = combined_df2["cosim_base_random"]
    combined_df = pd.concat([combined_df2, combined_df])
    # Create a box plot using Seaborn with 'hue' set to 'label'
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    # Use Seaborn's boxplot function with 'hue' set to 'label' to create side-by-side box plots
    sns.boxplot(data=combined_df, x="label", y='cosim_base_full_prompt', palette='Set1', width=0.6, linewidth=1.5, showfliers=False)
    # sns.boxplot(data=combined_df, y='cosim_base_random', color='grey', palette='Set1', showfliers=False, width=0.6, linewidth=1.5)

    # Customize the plot
    plt.title('Next Token Cosine Similarities')
    plt.xlabel('')
    plt.ylabel('Cosine Similarity')

    # Show the plot
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(f'figures/attributes_rank_cosim.png')

# slice = attribute_in_top_1_event, attributes_rate_bucket
def plots_sliced_by_extraction_rate(knowns_combined, mistakes_combined, slice_label):
    def agg_metrics_split_by_extraction(df, label):
        min_subject_only = df.groupby(['relation_before_subject', slice_label])['subject_only_attribute_rank'].min().reset_index()
        min_relation_only = df.groupby(['relation_before_subject', slice_label])['relation_before_subject_relation_only_attribute_rank'].min().reset_index()
        min_relation_before_subject = df.groupby(['relation_before_subject', slice_label])['relation_before_subject_attribute_rank'].min().reset_index()
        max_attributes_rate = df.groupby(['relation_before_subject', slice_label])['attributes_rate'].max().reset_index()
        min_pred_attribute_rank = df.groupby(['relation_before_subject', slice_label])['relation_before_subject_pred_attribute_rank'].min().reset_index()
        
        extraction_event = None
        if slice_label == "attributes_rate_bucket":
            df["attribute_in_top_1_event"] = df["attribute_in_top_1_event"].astype(int)
            extraction_event = df.groupby(['relation_before_subject', slice_label]).agg({'attribute_in_top_1_event': 'max'}).reset_index()

        min_subject_only["label"] = label
        min_relation_only["label"] = label
        min_relation_before_subject["label"] = label
        max_attributes_rate["label"] = label
        min_pred_attribute_rank["label"] = label
        if slice_label == "attributes_rate_bucket":
            extraction_event["label"] = label

        return min_subject_only, min_relation_only, min_relation_before_subject, max_attributes_rate, min_pred_attribute_rank, extraction_event

    knowns_min_subject_only, knowns_min_relation_only, knowns_min_relation_before_subject, knowns_max_attributes_rate, knowns_min_pred_attribute_rank, knowns_extraction_event = agg_metrics_split_by_extraction(knowns_combined, "correct")
    mistakes_min_subject_only, mistakes_min_relation_only, mistakes_min_relation_before_subject, mistakes_max_attributes_rate, mistakes_min_pred_attribute_rank, mistakes_extraction_event = agg_metrics_split_by_extraction(mistakes_combined, "mistakes")

    min_subject_only = pd.concat([mistakes_min_subject_only, knowns_min_subject_only])
    min_relation_only = pd.concat([mistakes_min_relation_only, knowns_min_relation_only])
    min_relation_before_subject = pd.concat([mistakes_min_relation_before_subject, knowns_min_relation_before_subject])
    max_attributes_rate = pd.concat([mistakes_max_attributes_rate, knowns_max_attributes_rate])
    min_pred_attribute_rank = pd.concat([mistakes_min_pred_attribute_rank, knowns_min_pred_attribute_rank])
    extraction_event = pd.concat([mistakes_extraction_event, knowns_extraction_event]) if slice_label == "attributes_rate_bucket" else None

    plt.clf()
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))

    # Plot "Minimum Subject Only Attribute Rank"
    sns.boxplot(
        x=slice_label,
        y="subject_only_attribute_rank",
        hue="label",
        data=min_subject_only,
        showfliers=False,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("Minimum Subject Only Attribute Rank")
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylim(bottom=1)


    # Plot "Minimum Relation Before Subject Attribute Rank"
    sns.boxplot(
        x=slice_label,
        y="relation_before_subject_attribute_rank",
        hue="label",
        data=min_relation_before_subject,
        showfliers=False,
        ax=axes[0, 1],
    )
    axes[0, 1].set_title("Minimum Relation Before Subject Attribute Rank")
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylim(bottom=1)

    # Plot "Minimum Relation Only Attribute Rank"
    sns.boxplot(
        x=slice_label,
        y="relation_before_subject_relation_only_attribute_rank",
        hue="label",
        data=min_relation_only,
        showfliers=False,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Minimum Relation Only Attribute Rank")
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_ylim(bottom=1)

    # Plot "Maximum Attributes Rate"
    sns.boxplot(
        x=slice_label,
        y="attributes_rate",
        hue="label",
        data=max_attributes_rate,
        showfliers=False,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Maximum Attributes Rate")

    # Plot "Prediction Attributes Rank"
    sns.boxplot(
        x=slice_label,
        y="relation_before_subject_pred_attribute_rank",
        hue="label",
        data=min_pred_attribute_rank,
        showfliers=False,
        ax=axes[2, 0]
    )
    axes[2, 0].set_yscale('log')
    axes[2, 0].set_title("Prediction Minimum Attribute Rank")

    if slice_label == "attributes_rate_bucket":
        # Plot "Extraction Event"
        extraction_rate = extraction_event.groupby(['label', slice_label])['attribute_in_top_1_event'].mean().reset_index()
        sns.barplot(data=extraction_rate, x=slice_label, y='attribute_in_top_1_event', palette="viridis", hue="label", ax=axes[2,1])
        axes[2, 1].set_ylabel("extraction_rate")
        axes[2, 1].set_title("Extraction Rate")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"/home/gamir/DER-Roei/dhgottesman/DFR/important/figures/{slice_label}_split_summary.png")

def bucket_attributes_rate(value):
    if value <= 0.05:
        return 0.05
    elif value <= 0.1:
        return 0.1
    elif value <= 0.25:
        return 0.25
    elif value <= .5:
        return .5
    elif value <= .75:
        return .75
    else:
        return 1.0

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

knowns_buckets = knowns_combined.groupby('relation_before_subject').agg({'attributes_rate': 'max'})
knowns_buckets = knowns_buckets.rename(columns={'attributes_rate': 'attributes_rate_bucket'})
knowns_buckets['attributes_rate_bucket'] = knowns_buckets['attributes_rate_bucket'].apply(bucket_attributes_rate)
knowns_combined = pd.merge(knowns_combined, knowns_buckets, on="relation_before_subject")

mistakes_buckets = mistakes_combined.groupby('relation_before_subject').agg({'attributes_rate': 'max'})
mistakes_buckets = mistakes_buckets.rename(columns={'attributes_rate': 'attributes_rate_bucket'})
mistakes_buckets['attributes_rate_bucket'] = mistakes_buckets['attributes_rate_bucket'].apply(bucket_attributes_rate)
mistakes_combined = pd.merge(mistakes_combined, mistakes_buckets, on="relation_before_subject")

knowns_buckets.value_counts()
mistakes_buckets.value_counts()

plots_sliced_by_extraction_rate(knowns_combined, mistakes_combined, "attributes_rate_bucket")

knowns_sample = knowns_combined.groupby(["subject", "attribute", "relation_id", "relation_before_subject"]).agg({"subject_only_attribute_rank": "min", "relation_before_subject_attribute_rank": "min", "relation_before_subject_relation_only_attribute_rank": "min", "relation_before_subject_pred_attribute_rank": "min", "next_25_tokens": "first", "attributes_rate": "max", "attribute_in_top_1_event": "first"})
knowns_sample.to_csv("important/flow_analysis/knowns_examples.csv")

mistakes_next_25 = load_df_from_dir("/home/gamir/DER-Roei/dhgottesman/DFR/important/flow_analysis/mistakes_examples")
mistakes_sample = pd.merge(mistakes_combined, mistakes_next_25, on=["relation_before_subject", "attribute"])
mistakes_sample = mistakes_sample.groupby(["subject", "attribute", "relation_id", "relation_before_subject"]).agg({"subject_only_attribute_rank": "min", "relation_before_subject_attribute_rank": "min", "relation_before_subject_relation_only_attribute_rank": "min", "relation_before_subject_pred_attribute_rank": "min", "next_25_tokens": "first", "attributes_rate": "max", "attribute_in_top_1_event": "first"})
mistakes_sample.to_csv("/home/gamir/DER-Roei/dhgottesman/DFR/important/flow_analysis/mistakes_examples/mistakes_examples.csv")


