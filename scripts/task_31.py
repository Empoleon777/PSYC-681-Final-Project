import pandas as pd
import random
import copy
import os
import re
import json
from difflib import SequenceMatcher

TARGET_MAP = {
    "corporations": "wealthy individuals",
    "immigrants": "citizens",
    "police": "teachers",
    "capitalism": "communism",
}

STANCE_RULES = [
    ("should", "should not"),
    ("is", "is not"),
    ("are", "are not"),
    ("too expensive", "worth the cost"),
    ("senseless", "necessary"),    
    ("reckless", "strategic"),
    ("stupid", "smart"),
    ("dumb", "smart"),
    ("foolish", "smart"),
    ("insane", "rational"),
]

FRAME_INSERTS = {
    "economic": "to improve economic growth",
    "fairness": "to ensure fairness",
    "efficiency": "to increase efficiency",
    "security": "to increase public safety",
}

CUE_WORDS_AND_PHRASES = [
    "morally", "disgusting", "unfair", "evil",
    "as a citizen", "as a taxpayer", "we must",
    "we should", "insane", "stupid", "dumb", 
    "foolish", "traitor", "asset", "shill"
]

ACTION_WORDS = ["should", "must", "need", "have to", "require"]

def replace_case_insensitive(text, old, new):
    return re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)

def apply_target_swap(text):
    items = list(TARGET_MAP.items())
    random.shuffle(items)
    for key, value in items:
        if key.lower() in text.lower():
            return replace_case_insensitive(text, key, value)
        
def apply_stance_reversal(text):
    items = sorted(STANCE_RULES, key=lambda x: -len(x[0]))
    random.shuffle(items)
    for key, value in items:
        if key.lower() in text.lower():
            if value.lower() in text.lower():
                continue
            return replace_case_insensitive(text, key, value)
        
def apply_frame_inserts(text):
    items = list(FRAME_INSERTS.values())
    random.shuffle(items)
    frame = items[0]
        
    if text.endswith("."):
        return text[:-1] + f", {frame}."
    else:
        return frame + " " + text
        
def remove_cue_words(text):
    cues = copy.deepcopy(CUE_WORDS_AND_PHRASES)
    random.shuffle(cues)
    for cue in cues:
        if cue.lower() in text.lower():
            removed = replace_case_insensitive(text, cue, "")
            removed = re.sub(rf"\b{re.escape(cue)}\b", "", removed, flags=re.IGNORECASE)
            removed = re.sub(r"\s+", " ", removed).strip()
            return removed

def get_expected_direction(edit_type):
    mapping = {
        "target_swap": {"effect": "ideology_stable"},
        "stance_reversal": {"econ": "flip", "social": "flip", "intensity": "same_or_lower"},
        "frame_change": {"effect": "frame_shift_only"},
        "cue_removal": {"effect": "lower_intensity_lower_moralization"}
    }
    return json.dumps(mapping[edit_type])

def build_row(row, edited_text, edit_type):
    return {
        "post_id": row["post_id"],
        "original_text": row["text"],
        "edited_text": edited_text,
        "edit_type": edit_type,
        "expected_direction": get_expected_direction(edit_type)
    }

def edit_distance_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def make_counterfactuals(df, n_per_type=50):
    rows = []
    random.seed(42)

    counts = {
        "target_swap": 0,
        "stance_reversal": 0,
        "frame_change": 0,
        "cue_removal": 0
    }

    types_needed = {
        "target_swap": n_per_type,
        "stance_reversal": n_per_type,
        "frame_change": n_per_type,
        "cue_removal": n_per_type,
    }

    shuffled = df.sample(frac=1, random_state=42)

    for _, row in shuffled.iterrows():
        text = row["text"]

        if counts["target_swap"] < n_per_type:
            edited = apply_target_swap(text)
            if edited:
                ratio = edit_distance_ratio(text, edited)
                if 0.6 < ratio < 0.98:
                    rows.append(build_row(row, edited, "target_swap"))
                    counts["target_swap"] += 1
                    continue

        if counts["stance_reversal"] < n_per_type:
            edited = apply_stance_reversal(text)
            if edited:
                ratio = edit_distance_ratio(text, edited)
                if 0.6 < ratio < 0.98:
                    rows.append(build_row(row, edited, "stance_reversal"))
                    counts["stance_reversal"] += 1
                    continue

        if counts["frame_change"] < n_per_type:
            edited = apply_frame_inserts(row["text"])
            if edited: 
                ratio = edit_distance_ratio(row["text"], edited)
                if 0.6 < ratio < 0.98:
                    row = build_row(row, edited, "frame_change")
                    rows.append(row)
                    counts["frame_change"] += 1                    
                    continue

        if counts["cue_removal"] < n_per_type:
            edited = remove_cue_words(row["text"])
            if edited:
                ratio = edit_distance_ratio(row["text"], edited)
                if 0.6 < ratio < 0.98:
                    row = build_row(row, edited, "cue_removal")
                    rows.append(row)
                    counts["cue_removal"] += 1
                    continue
    
    print(f"Edits: {counts}")
    cf_set = pd.DataFrame(rows)
    os.makedirs('Data/counterfactuals', exist_ok=True)
    cf_set.to_csv("Data/counterfactuals/cf_set.csv", index=False)
    return pd.DataFrame(rows)

def main():
    # gold_annotations.csv carries per-annotator labels but no text; pull text
    # from raw_posts and keep one row per post_id.
    raw_posts = pd.read_csv("outputs/ingestion/raw_posts_60k.csv")
    gold = pd.read_csv("outputs/annotation_60k/gold_annotations.csv")
    annotated_post_ids = gold["post_id"].drop_duplicates()
    data = raw_posts.merge(annotated_post_ids, on="post_id", how="inner")[["post_id", "text"]]

    cf = make_counterfactuals(data)

    print(len(cf))
    if len(cf) > 0:
        print(cf["edit_type"].value_counts())
        print("\nSample:")
        print(cf.sample(min(10, len(cf)))[["edit_type", "original_text", "edited_text"]])


if __name__ == "__main__":
    main()