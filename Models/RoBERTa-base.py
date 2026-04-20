import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments


class MultiTaskRoberta(nn.Module):
    def __init__(self, model_name, num_domains=5, num_facets=12):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.domain_head = nn.Linear(hidden_size, num_domains)
        self.facet_head = nn.Linear(hidden_size, num_facets)
        self.ideology_head = nn.Linear(hidden_size, num_facets * 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)

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
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        domain_labels = inputs.pop("domain_labels")
        facet_labels = inputs.pop("facet_labels")
        ideology_labels = inputs.pop("ideology_labels")

        outputs = model(**inputs)

        bce = nn.BCEWithLogitsLoss()
        ce = nn.CrossEntropyLoss(reduction='none')

        domain_loss = bce(outputs["domain_logits"], domain_labels.float())
        facet_loss = bce(outputs["facet_logits"], facet_labels.float())
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

training_args = TrainingArguments(
    output_dir="./roberta-mitweet",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

model = MultiTaskRoberta(model_name)

trainer = MultiTaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=valid_data,
)

print(train_data.column_names)
print(train_data[0])

# print(train_data[0])
# print(dataset["train"][0])

# trainer.train()