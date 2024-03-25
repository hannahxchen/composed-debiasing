import os
import copy
import pickle
import argparse
import pandas as pd
import re, json
from tqdm import tqdm
from random import sample, choices
from datasets import load_dataset, concatenate_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir", type=str, default=None, help="dataset directory"
    )

    parser.add_argument("--output_dir", type=str, default=None, required=True, help="Where to store the output.")
    args = parser.parse_args()

    return args

def lookup_stats(job_gender_stats, title, gender):
        return job_gender_stats.loc[(job_gender_stats.title == title) & (job_gender_stats.gender == gender)]["count"].item()


def get_idxs(df, title, gender):
    return df.loc[(df.title == title) & (df.gender == gender)].idx.tolist()


def get_data_by_idx(dataset, sampled_idxs):
    sampled = []
    for idx in sampled_idxs:
        sampled.append(dataset["train"][idx])
    return sampled


def cda(dataset):
    orig_train = copy.deepcopy(dataset)
    orig_train = orig_train.rename_column("title_scrubbed_gender_swapped", "temp")
    orig_train = orig_train.rename_column("title_scrubbed", "title_scrubbed_gender_swapped")
    orig_train = orig_train.rename_column("temp", "title_scrubbed")
    data = concatenate_datasets([orig_train, dataset])
    return data


def save(args, data, save_filename):
    with open(os.path.join(args.data_dir, save_filename), "w") as f:
        for x in data:
            json.dump(x, f)
            f.write("\n")


def main():
    args = parse_args()

    data_files = {"train": os.path.join(args.data_dir, "train.json")}
    dataset = load_dataset("json", data_files=data_files)

    # # CDA
    # data = cda(dataset["train"])
    # data = data.remove_columns(["raw", "raw_title", "start_pos", "name", "gender_title_scrubbed"])
    # save(args, data, "train_cda.json")

    dataset = dataset.remove_columns(["raw", "raw_title", "start_pos", "name", "gender_title_scrubbed"])
    df = pd.DataFrame({"idx": list(range(len(dataset["train"]))), "title": dataset["train"]["title"], "gender": dataset["train"]["gender"]})
    job_gender_stats = df.groupby(["title", "gender"]).count().reset_index()
    job_gender_stats.columns = ["title", "gender", "count"]


    all_titles = df.title.unique().tolist()

    # subsampling
    sampled_idxs = []

    for title in all_titles:
        m_counts = lookup_stats(job_gender_stats, title, "M")
        f_counts = lookup_stats(job_gender_stats, title, "F")

        m_examples = get_idxs(df, title, "M")
        f_examples = get_idxs(df, title, "F")

        if m_counts == f_counts:
            sampled_idxs.extend(m_examples)
            sampled_idxs.extend(f_examples)

        elif m_counts < f_counts:
            sampled_idxs.extend(m_examples)
            sample_size = m_counts
            sampled_idxs.extend(sample(f_examples, sample_size))
            
        elif f_counts < m_counts:
            sampled_idxs.extend(f_examples)
            sample_size = f_counts
            sampled_idxs.extend(sample(m_examples, sample_size))

    data = get_data_by_idx(dataset, sampled_idxs)
    save(args, data, "train_subsampled.json")

    # oversampling
    sampled_idxs = []

    for title in all_titles:
        m_counts = lookup_stats(title, "M")
        f_counts = lookup_stats(title, "F")

        m_examples = get_idxs(title, "M")
        f_examples = get_idxs(title, "F")

        sampled_idxs.extend(m_examples)
        sampled_idxs.extend(f_examples)

        if m_counts == f_counts:
            continue

        elif m_counts < f_counts:
            sample_size = f_counts - m_counts
            sampled_idxs.extend(choices(m_examples, k=sample_size))
            
        elif f_counts < m_counts:
            sample_size = m_counts - f_counts
            sampled_idxs.extend(choices(f_examples, k=sample_size))

    data = get_data_by_idx(dataset, sampled_idxs)
    save(args, dataset, sampled_idxs, "train_oversampled.json")

    # # subsampling + CDA
    # data_files = {"train": os.path.join(args.data_dir, "train_subsampled.json")}
    # dataset = load_dataset("json", data_files=data_files)

    # data = cda(dataset["train"])
    # save(args, data, "train_subsampled_cda.json")

    # # oversampling + CDA
    # data_files = {"train": os.path.join(args.data_dir, "train_oversampled.json")}
    # dataset = load_dataset("json", data_files=data_files)

    # data = cda(dataset["train"])
    # save(args, data, "train_oversampled_cda.json")