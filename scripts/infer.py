from resnet_train import TestData, DEVICE, CLASSES
from pathlib import Path
import torchvision.models as models
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_model():
    # model = models.resnet18()
    # model.conv1 = nn.Conv2d(
    #     1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    # )
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 10, bias=True)

    # === Efficientnet ===
    model = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_efficientnet_b0",
        pretrained=False,
    )
    model.stem.conv = nn.Conv2d(
        1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )
    num_ftrs = model.classifier.fc.in_features
    model.classifier.fc = nn.Linear(num_ftrs, 10, bias=True)

    return model


def load_efficientnet():
    model = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_efficientnet_b0",
        pretrained=False,
    )
    model.stem.conv = nn.Conv2d(
        1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    )
    num_ftrs = model.classifier.fc.in_features
    model.classifier.fc = nn.Linear(num_ftrs, 10, bias=True)

    return model


def load_checkpoint(path: Path):
    model = load_model()
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    test_dir = PROJECT_DIR / "hackaton_ds/test"
    test_markup = PROJECT_DIR / "hackaton_ds/submission_xvector_cos_sim.csv"
    test = TestData(test_dir, test_markup)
    # test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)

    model = load_checkpoint("models/efficientnet_70ep.pt").to(DEVICE)
    model.eval()

    test_answers = list()
    test_loss = list()
    for X, Y in tqdm(test):
        X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        test_answers.extend(torch.argmax(preds, dim=-1).cpu().data.numpy())

    submission = pd.read_csv(test_markup, engine="python")
    submission["category"] = pd.Series(test_answers).apply(lambda x: CLASSES[x])
    submission.to_csv(PROJECT_DIR / "hackaton_ds/submission.csv", index=None)
