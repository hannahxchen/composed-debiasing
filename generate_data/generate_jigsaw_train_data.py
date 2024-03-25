import os
import copy
import pickle
import argparse
import pandas as pd
import re, json
from tqdm import tqdm
from random import sample, choices

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default=None, help="dataset directory"
    )

    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the output.")
    args = parser.parse_args()

    return args

def save(args, data, save_filename):
    data.to_csv(os.path.join(args.data_dir, save_filename), index=False)


def subsample_ids(jigsaw_train_df, cda=False):
    sampled_ids = []
    if cda:
        gender_column = "swapped_gender"
    else:
        gender_column = "gender"

    for label in [1, 0]:
        m_example_ids = jigsaw_train_df[(jigsaw_train_df[gender_column] == "M") & (jigsaw_train_df.label == label)].id.tolist()
        f_example_ids = jigsaw_train_df[(jigsaw_train_df[gender_column] == "F") & (jigsaw_train_df.label == label)].id.tolist()

        m_counts = len(m_example_ids)
        f_counts = len(f_example_ids)

        if m_counts < f_counts:
            sampled_ids.extend(m_example_ids)
            sample_size = m_counts
            sampled_ids.extend(sample(f_example_ids, sample_size))
            
        elif f_counts < m_counts:
            sampled_ids.extend(f_example_ids)
            sample_size = f_counts
            sampled_ids.extend(sample(m_example_ids, sample_size))

    return sampled_ids


def oversample_ids(jigsaw_train_df, cda=False):
    oversampled_ids = []
    if cda:
        gender_column = "swapped_gender"
    else:
        gender_column = "gender"
 
    for label in [1, 0]:
        m_example_ids = jigsaw_train_df[(jigsaw_train_df[gender_column] == "M") & (jigsaw_train_df.label == label)].id.tolist()
        f_example_ids = jigsaw_train_df[(jigsaw_train_df[gender_column] == "F") & (jigsaw_train_df.label == label)].id.tolist()


        m_counts = len(m_example_ids)
        f_counts = len(f_example_ids)

        if m_counts < f_counts:
            sample_size = f_counts - m_counts
            oversampled_ids.extend(choices(m_example_ids, k=sample_size))
            
        elif f_counts < m_counts:
            sample_size = m_counts - f_counts
            oversampled_ids.extend(choices(f_example_ids, k=sample_size))
            
    return oversampled_ids


def main():
    args = parse_args()
    jigsaw_train_df = pd.read_csv(os.path.join(args.dataset_dir, "processed_train.csv"))

    # subsampling
    subsampled_ids = subsample_ids(jigsaw_train_df)
    data = jigsaw_train_df[jigsaw_train_df.id.isin(subsampled_ids)]
    save(args, data, "train_subsampled.csv")

    # oversampling
    oversampled_ids = oversample_ids(jigsaw_train_df)
    data = jigsaw_train_df.copy()

    for id in oversampled_ids:
        new_row = jigsaw_train_df.loc[jigsaw_train_df.id == id].values.flatten().tolist()
        data.loc[len(data.index)] = new_row
    save(args, data, "train_oversampled.csv")

    # subsampling + CDA
    sampled_ids = subsample_ids(jigsaw_train_df)
    cda_sampled_ids = subsample_ids(jigsaw_train_df, cda=True)
    sampled_data = jigsaw_train_df[jigsaw_train_df.id.isin(sampled_ids)]
    cda_sampled_data = jigsaw_train_df[jigsaw_train_df.id.isin(cda_sampled_ids)]
    cda_sampled_data = cda_sampled_data.rename(columns={
        'comment_text': 'gender_swapped_comment', 
        'gender_swapped_comment': 'comment_text',
        'gender': 'swapped_gender', 'swapped_gender': 'gender'
        })
    data = pd.concat((sampled_data, cda_sampled_data))
    data = data[["id", "target", "comment_text", "male", "female", "label", "gender", "gender_swapped_comment"]]
    save(args, data, "train_subsampled_cda.csv")


    # oversampling + CDA
    sampled_ids = oversample_ids(jigsaw_train_df)
    cda_sampled_ids = oversample_ids(jigsaw_train_df, cda=True)
    sampled_data = jigsaw_train_df.copy()
    for id in sampled_ids:
        new_row = jigsaw_train_df.loc[jigsaw_train_df.id == id].values.flatten().tolist()
        sampled_data.loc[len(sampled_data.index)] = new_row

    cda_sampled_data = jigsaw_train_df.copy()
    for id in cda_sampled_ids:
        new_row = jigsaw_train_df.loc[jigsaw_train_df.id == id].values.flatten().tolist()
        cda_sampled_data.loc[len(cda_sampled_data.index)] = new_row

    cda_sampled_data = cda_sampled_data.rename(columns={
        'comment_text': 'gender_swapped_comment', 
        'gender_swapped_comment': 'comment_text',
        'gender': 'swapped_gender', 'swapped_gender': 'gender'
        })
    data = pd.concat((sampled_data, cda_sampled_data))
    data = data[["id", "target", "comment_text", "male", "female", "label", "gender", "gender_swapped_comment"]]
    save(args, data, "train_oversampled_cda.csv")