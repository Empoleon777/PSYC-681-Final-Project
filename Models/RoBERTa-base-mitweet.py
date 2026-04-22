import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score

class MultiTaskRoberta(nn.Module):
    def __init__(self, model_name, num_domains=5, num_facets=12):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.domain_head = nn.Linear(hidden_size, num_domains)
        self.facet_head = nn.Linear(hidden_size, num_facets)
        self.ideology_head = nn.Linear(hidden_size, num_facets * 3)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]

        domain_logits = self.domain_head(pooled)
        facet_logits = self.facet_head(pooled)

        ideology_logits = self.ideology_head(pooled)
        ideology_logits = ideology_logits.view(-1, 12, 3)

        return {
            "domain_logits": domain_logits,
            "facet_logits": facet_logits,
            "ideology_logits": ideology_logits
        }

class MultiTaskTrainer(Trainer):
    def __init__(self, *args, pos_weight_domain=None, pos_weight_facet=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_domain = pos_weight_domain
        self.pos_weight_facet = pos_weight_facet
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        domain_labels = inputs.get("domain_labels")
        facet_labels = inputs.get("facet_labels")
        ideology_labels = inputs.get("ideology_labels")

        if domain_labels is not None:
            inputs.pop("domain_labels")
        if facet_labels is not None:
            inputs.pop("facet_labels")
        if ideology_labels is not None:
            inputs.pop("ideology_labels")

        outputs = model(**inputs)

        facet_labels = facet_labels.float()

        bce_domain = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight_domain.to(domain_labels.device)
        )

        bce_facet = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight_facet.to(facet_labels.device)
        )
        
        ce = nn.CrossEntropyLoss(reduction='none')

        domain_loss = bce_domain(outputs["domain_logits"], domain_labels.float())
        facet_loss = bce_facet(outputs["facet_logits"], facet_labels.float())
        mask = (ideology_labels != -1) & (facet_labels == 1)
        targets = ideology_labels.clamp(min=0).long()

        ce_loss = ce(
            outputs["ideology_logits"].view(-1, 3),
            targets.view(-1)
        ).view(-1, 12)

        ideology_loss = (ce_loss * mask).sum() / mask.sum().clamp(min=1)

        loss = domain_loss + facet_loss + ideology_loss

        return (loss, outputs) if return_outputs else loss

def tokenize(example):
    return tokenizer(
        example["tweet"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def add_labels(example):
    return {
        "domain_labels": [example[f"R{i}"] for i in range(1, 6)],

        "facet_labels": [
            example["R1-1-1"],
            example["R2-1-2"],
            example["R3-2-1"],
            example["R4-2-2"],
            example["R5-3-1"],
            example["R6-3-2"],
            example["R7-3-3"],
            example["R8-4-1"],
            example["R9-4-2"],
            example["R10-5-1"],
            example["R11-5-2"],
            example["R12-5-3"],
        ],

        "ideology_labels": [example[f"I{i}"] for i in range(1, 13)],
    }

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_multilabel_f1(logits, labels, threshold=0.5):
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    micro = f1_score(labels, preds, average="micro", zero_division=0)
    macro = f1_score(labels, preds, average="macro", zero_division=0)

    return micro, macro

def compute_ideology_f1(logits, labels, facet_labels, facet_logits):
    preds = logits.argmax(axis=-1)
    facet_probs = sigmoid(facet_logits)
    facet_preds = (facet_probs >= 0.5).astype(int)
    preds[facet_preds == 0] = -1

    mask = (labels != -1) & (facet_labels == 1)

    y_true = labels[mask]
    y_pred = preds[mask]

    if len(y_true) == 0:
        return 0.0, 0.0

    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return micro, macro

def compute_all_f1(outputs, labels):
    domain_logits = outputs["domain_logits"]
    facet_logits = outputs["facet_logits"]
    ideology_logits = outputs["ideology_logits"]

    domain_labels = labels["domain_labels"]
    facet_labels = labels["facet_labels"]
    ideology_labels = labels["ideology_labels"]

    d_micro, d_macro = compute_multilabel_f1(domain_logits, domain_labels)
    f_micro, f_macro = compute_multilabel_f1(facet_logits, facet_labels)
    i_micro, i_macro = compute_ideology_f1(
        ideology_logits, ideology_labels, facet_labels
    )

    return {
        "domain_micro_f1": d_micro,
        "domain_macro_f1": d_macro,
        "facet_micro_f1": f_micro,
        "facet_macro_f1": f_macro,
        "ideology_micro_f1": i_micro,
        "ideology_macro_f1": i_macro,
    }

def compute_metrics(eval_pred):
    global domain_thresholds, facet_thresholds

    logits, labels = eval_pred

    domain_logits, facet_logits, ideology_logits = logits

    domain_labels = labels[:, :5]
    facet_labels = labels[:, 5:17]
    ideology_labels = labels[:, 17:]

    if domain_thresholds is None:
        domain_thresholds = tune_thresholds(domain_logits, domain_labels)
        facet_thresholds = tune_thresholds(facet_logits, facet_labels)

    domain_probs = 1 / (1 + np.exp(-domain_logits))
    facet_probs = 1 / (1 + np.exp(-facet_logits))

    domain_preds = (domain_probs >= domain_thresholds).astype(int)
    facet_preds = (facet_probs >= facet_thresholds).astype(int)

    ideology_preds = ideology_logits.argmax(axis=-1)

    ideology_preds[facet_preds == 0] = -1

    mask = (ideology_labels != -1) & (facet_labels == 1)

    y_true = ideology_labels[mask]
    y_pred = ideology_preds[mask]

    i_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    i_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "domain_micro_f1": f1_score(domain_labels, domain_preds, average="micro", zero_division=0),
        "domain_macro_f1": f1_score(domain_labels, domain_preds, average="macro", zero_division=0),
        "facet_micro_f1": f1_score(facet_labels, facet_preds, average="micro", zero_division=0),
        "facet_macro_f1": f1_score(facet_labels, facet_preds, average="macro", zero_division=0),
        "ideology_micro_f1": i_micro,
        "ideology_macro_f1": i_macro,
    }

def compute_pos_weights(dataset):
    domain = np.stack(dataset["domain_labels"])
    facet = np.stack(dataset["facet_labels"])

    # avoid divide-by-zero
    domain_pos = domain.sum(axis=0)
    domain_neg = domain.shape[0] - domain_pos
    domain_weight = domain_neg / np.clip(domain_pos, 1, None)

    facet_pos = facet.sum(axis=0)
    facet_neg = facet.shape[0] - facet_pos
    facet_weight = facet_neg / np.clip(facet_pos, 1, None)

    return torch.tensor(domain_weight, dtype=torch.float32), \
           torch.tensor(facet_weight, dtype=torch.float32)

def tune_thresholds(logits, labels):
    probs = 1 / (1 + np.exp(-logits))
    num_labels = labels.shape[1]

    best_thresholds = np.zeros(num_labels)

    for j in range(num_labels):
        best_f1 = 0
        best_t = 0.5

        for t in np.linspace(0.05, 0.95, 19):
            preds = (probs[:, j] >= t).astype(int)
            f1 = f1_score(labels[:, j], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds[j] = best_t

    return best_thresholds

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("csv", data_files="Data/MITweet.csv")["train"]

dataset = dataset.map(add_labels)
dataset = dataset.map(tokenize)

dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_data = dataset["train"]
test_data = dataset["test"]

train_valid = train_data.train_test_split(test_size=0.25)
train_data = train_valid["train"]
valid_data = train_valid["test"]

train_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask",
             "domain_labels", "facet_labels", "ideology_labels"]
)

valid_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask",
             "domain_labels", "facet_labels", "ideology_labels"]
)

test_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask",
             "domain_labels", "facet_labels", "ideology_labels"]
)

pos_weight_domain, pos_weight_facet = compute_pos_weights(train_data)

training_args = TrainingArguments(
    output_dir="./roberta-mitweet",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    remove_unused_columns=False
)

model = MultiTaskRoberta(model_name)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    compute_metrics=compute_metrics,
    pos_weight_domain=pos_weight_domain,
    pos_weight_facet=pos_weight_facet,
)

# print(train_data[0])
# print(train_data.column_names)

# print(train_data[0])
# print(dataset["train"][0])

domain_thresholds = None
facet_thresholds = None

trainer.train()