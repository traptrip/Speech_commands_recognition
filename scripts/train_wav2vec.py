import random
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os

import torch.nn.functional as F
import torch.optim as opt
import torchaudio
import torchaudio.transforms as T
import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from datasets import load_dataset, load_metric
from transformers import AutoFeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer


SEED = 1234


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
    )
    return inputs


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


seed_everything(SEED)

DATA_PATH = Path("../data")
WEIGHTS_PATH = Path("OthmaneJ/distil-wav2vec2")
EXP_NAME = WEIGHTS_PATH.name

max_duration = 1.0  # seconds
MAX_AUDIO_LEN = 16000  # в отсчётах sr

batch_size = 8
DEVICE = "cuda"
N_EPOCHS = 2


dataset = load_dataset("superb", "ks")
metric = load_metric("accuracy")

feature_extractor = AutoFeatureExtractor.from_pretrained(WEIGHTS_PATH)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

encoded_dataset = dataset.map(
    preprocess_function, remove_columns=["audio", "file"], batched=True
)


wav2vec = AutoModelForAudioClassification.from_pretrained(
    WEIGHTS_PATH,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label,
)
args = TrainingArguments(
    f"pretrained_models/{EXP_NAME}-finetuned-ks",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=N_EPOCHS,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    wav2vec,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)
trainer.train()
