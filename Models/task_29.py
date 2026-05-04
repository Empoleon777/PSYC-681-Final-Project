import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_in_domain(df, seed=42):
    train, temp = train_test_split(df, test_size=0.3, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, random_state=seed)

    return {"train": train, "val": val, "test": test}

def check_no_overlap(a, b):
    return set(a["post_id"]).isdisjoint(set(b["post_id"]))

def split_loto(df):
    splits = {}

    for topic in df["topic"].unique():
        test = df[df["topic"] == topic]
        train = df[df["topic"] != topic]

        splits[topic] = {
            "train": train,
            "test": test
        }

    return splits

def split_loco(df):
    splits = {}

    for comm in df["community"].unique():
        test = df[df["community"] == comm]
        train = df[df["community"] != comm]

        splits[comm] = {
            "train": train,
            "test": test
        }

    return splits

def split_external(df):
    train = df[df["source"] == "internal"]
    test = df[df["source"] == "external"]

    return {"train": train, "test": test}

def save_split(split, name):
    os.makedirs('../Data/splits', exist_ok=True)
    for part, df_part in split.items():
        df_part.to_csv(f"../Data/splits/{name}_{part}.csv", index=False)

def safe_name(s):
    return str(s).replace(" ", "_").replace("/", "_")

def build_splits(df, seed=42):
    splits = {}

    splits["in_domain"] = split_in_domain(df, seed)
    assert check_no_overlap(splits["in_domain"]["train"], splits["in_domain"]["val"])
    assert check_no_overlap(splits["in_domain"]["train"], splits["in_domain"]["test"])
    assert check_no_overlap(splits["in_domain"]["val"], splits["in_domain"]["test"])
    splits["loto"] = split_loto(df)
    for topic in df["topic"].unique():
        assert check_no_overlap(
            splits["loto"][topic]["train"],
            splits["loto"][topic]["test"]
        )
    splits["loco"] = split_loco(df)
    for comm in df["community"].unique():
        assert check_no_overlap(
            splits["loco"][comm]["train"],
            splits["loco"][comm]["test"]
        )
    splits["external"] = split_external(df)
    assert check_no_overlap(splits["external"]["train"], splits["external"]["test"])

    save_split(splits["in_domain"], "Domain")

    for key, split in splits["loto"].items():
        save_split(split, f"loto_{key}")

    for key, split in splits["loco"].items():
        save_split(split, f"loco_{key}")

    save_split(splits["external"], "Transfer")

    return splits