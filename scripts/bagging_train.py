from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from efficientnet_train import CLASSES, DEVICE
from efficientnet_train import TrainData as EfficientnetTrainData
from efficientnet_train import TestData
from efficientnet_train import load_model as load_efficientnet_model
from resnet_train import TrainData as ResnetTrainData
from resnet_train import load_model as load_resnet_model

PROJECT_DIR = Path(__file__).resolve().parents[1]
softmax = nn.Softmax(dim=0)


def resnet_train_infer(model):
    noise_dir = PROJECT_DIR / "hackaton_ds/noises"
    train_dir = PROJECT_DIR / "hackaton_ds/train"
    train = ResnetTrainData(train_dir, noise_dir)
    train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0)

    train_answers = list()
    for X, _ in tqdm(train_loader):
        # X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        train_answers.append(logists)

    return np.asfarray(train_answers), train.classes


def efficientnet_train_infer(model):
    noise_dir = PROJECT_DIR / "hackaton_ds/noises"
    train_dir = PROJECT_DIR / "hackaton_ds/train"
    train = EfficientnetTrainData(train_dir, noise_dir)
    train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0)

    train_answers = list()
    for X, _ in tqdm(train_loader):
        # X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        train_answers.append(logists)

    return np.asfarray(train_answers), train.classes


if __name__ == "__main__":
    # test_dir = PROJECT_DIR / "hackaton_ds/test"
    # test_markup = PROJECT_DIR / "hackaton_ds/submission_xvector_cos_sim.csv"
    # test = TestData(test_dir, test_markup)
    # test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    resnet = load_resnet_model().to(DEVICE)
    resnet.load_state_dict(
        torch.load(PROJECT_DIR / "models/Resnet_16/resnet16_95ep.pt")
    )
    resnet.eval()

    efficientnet = load_efficientnet_model().to(DEVICE)
    efficientnet.load_state_dict(
        torch.load(PROJECT_DIR / "models/Efficientnet/efficientnet_70ep.pt")
    )
    efficientnet.eval()

    resnet_train_answers, classes1 = resnet_train_infer(resnet)
    classes = [CLASSES.index(class_name) for class_name in classes1]
    efficientnet_train_answers, _ = efficientnet_train_infer(efficientnet)
    train_answers = np.concatenate(
        (resnet_train_answers, efficientnet_train_answers), axis=1
    )

    clf = LogisticRegression(random_state=0, solver="saga").fit(train_answers, classes)

    # now you can save it to a file
    with open(str(PROJECT_DIR / "models/log_reg.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # resnet_test_answers = test_infer(resnet, test)
    # print(resnet_test_answers)
    # efficientnet_test_answers = test_infer(efficientnet, test)
    # print(efficientnet_test_answers)
    # test_answers = np.concatenate(
    #     (resnet_test_answers, efficientnet_test_answers), axis=1
    # )
    # print(test_answers)
