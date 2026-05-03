import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class Flat(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size
        self.relevance_head = nn.Linear(hidden_size, 2)
        self.econ_head = nn.Linear(hidden_size, 3)
        self.social_head = nn.Linear(hidden_size, 3)
    
    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = outputs.last_hidden_state[:, 0]

        relevance_logits = self.relevance_head(pooled)
        econ_logits = self.econ_head(pooled)
        social_logits = self.social_head(pooled)

        return {
            "relevance_logits": relevance_logits,
            "econ_logits": econ_logits,
            "social_logits": social_logits
        }
    
class MultiTaskTrainer(Trainer):
    def __init__(self, *args, pos_weight_relevance=None, pos_weight_econ=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_relevance = pos_weight_relevance
        self.pos_weight_econ = pos_weight_econ
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        relevance_labels = inputs.get("relevance_labels")
        econ_labels = inputs.get("econ_labels")
        social_labels = inputs.get("social_labels")

        if relevance_labels is not None:
            inputs.pop("relevance_labels")
        if econ_labels is not None:
            inputs.pop("econ_labels")
        if social_labels is not None:
            inputs.pop("social_labels")

        outputs = model(**inputs)

        econ_labels = econ_labels.float()

        ce = nn.CrossEntropyLoss(ignore_index=-1)

        relevance_loss = ce(outputs["relevance_logits"], relevance_labels.long())
        econ_loss = ce(outputs["econ_logits"], econ_labels.long())
        social_loss = ce(outputs["social_logits"], social_labels.long())

        loss = relevance_loss + econ_loss + social_loss

        return (loss, outputs) if return_outputs else loss
    
class VisualLoggerCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_world_process_zero:
            print(f"Metrics: \n{metrics}")

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

def add_labels(example):
    return {
        "relevance_labels": example["q01_relevance"],
        "econ_labels": example["q07_economic_direction"],
        "social_labels": example["q08_social_direction"],
    }

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    relevance_logits, econ_logits, social_logits = logits

    relevance_labels = labels[:, 0]
    econ_labels = labels[:, 1]
    social_labels = labels[:, 2]

    econ_preds = np.argmax(econ_logits, axis=-1)
    social_preds = np.argmax(social_logits, axis=-1)

    return {
        "econ_macro_f1": f1_score(econ_labels, econ_preds, average="macro", zero_division=0),
        "social_macro_f1": f1_score(social_labels, social_preds, average="macro", zero_division=0),
    }

seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
    
dataset = load_dataset("csv", data_files=r"outputs/annotation_60k/gold_annotations.csv")
dataset = dataset["train"]
dataset = dataset.map(add_labels)
dataset = dataset.map(tokenize)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_data = dataset["train"]
test_data = dataset["test"]
train_valid = train_data.train_test_split(test_size=0.1)
train_data = train_valid["train"]
valid_data = train_valid["test"]

train_data.set_format(
    type="torch",
    columns=[
        "input_ids", "attention_mask",
        "relevance_labels", "econ_labels", "social_labels"
    ]
)

valid_data.set_format(
    type="torch",
    columns=[
        "input_ids", "attention_mask",
        "relevance_labels", "econ_labels", "social_labels"
    ]
)

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

model = Flat(model_name)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
    compute_metrics=compute_metrics,
    callbacks=[VisualLoggerCallback()]
)

trainer.train()