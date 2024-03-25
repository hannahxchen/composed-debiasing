import os
import pandas as pd
from gender_bender import gender_bend
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
tqdm.pandas()

DATASET_DIR = "datasets/jigsaw-toxicity/orig_data"
OUTPUT_DIR = "datasets/jigsaw-toxicity"


def jigsaw_preprocess(jigsaw_df):
    if "target" in jigsaw_df.columns.tolist():
        toxicity_col_name = "target"
    else:
        toxicity_col_name = "toxicity"
    jigsaw_df = jigsaw_df.fillna(0)

    jigsaw_df["diff"] = jigsaw_df.female - jigsaw_df.male
    jigsaw_df = jigsaw_df[jigsaw_df["diff"].abs() > 0.5]
    jigsaw_df = jigsaw_df[["id", toxicity_col_name, "comment_text", "male", "female"]]

    jigsaw_df["label"] = jigsaw_df[toxicity_col_name].apply(lambda x: 1 if x > 0.5 else 0)
    jigsaw_df["gender"] = jigsaw_df.female > jigsaw_df.male
    jigsaw_df["gender"] = jigsaw_df.gender.apply(lambda x: "F" if x == 1 else "M")

    return jigsaw_df

if __name__ == "__main__":
    jigsaw_train_df = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
    jigsaw_test_df = pd.read_csv(os.path.join(DATASET_DIR, "test_public_expanded.csv"))
    jigsaw_test_df2 = pd.read_csv(os.path.join(DATASET_DIR, "test_private_expanded.csv"))
    df_test = pd.concat((jigsaw_test_df, jigsaw_test_df2))

    jigsaw_train_df = jigsaw_preprocess(jigsaw_train_df)
    df_test = jigsaw_preprocess(df_test)

    jigsaw_train_df["gender_swapped_comment"] = jigsaw_train_df.comment_text.progress_apply(gender_bend)
    df_test["gender_swapped_comment"] = df_test.comment_text.progress_apply(gender_bend)

    df_train, df_valid = train_test_split(jigsaw_train_df, test_size=0.2)

    df_train.to_csv(os.path.join(OUTPUT_DIR, "processed_train.csv"), index=False)
    df_valid.to_csv(os.path.join(OUTPUT_DIR, "processed_valid.csv"), index=False)
    df_test.to_csv(os.path.join(OUTPUT_DIR, "processed_test.csv"), index=False)