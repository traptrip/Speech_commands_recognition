from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from tqdm import tqdm

from efficientnet_train import CLASSES, DEVICE
from efficientnet_train import TestData
from efficientnet_train import load_model as load_efficientnet_model
from resnet_train import load_model as load_resnet_model

PROJECT_DIR = Path(__file__).resolve().parents[1]
softmax = nn.Softmax(dim=0)


def test_infer(model, test):
    test_answers = list()
    for X, _ in tqdm(test):
        X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        test_answers.append(logists)

    return np.asfarray(test_answers)


if __name__ == "__main__":
    test_dir = PROJECT_DIR / "hackaton_ds/test"
    test_markup = PROJECT_DIR / "hackaton_ds/submission_xvector_cos_sim.csv"
    test = TestData(test_dir, test_markup)

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

    with open(str(PROJECT_DIR / "models/log_reg.pkl"), "rb") as f:
        clf = pickle.load(f)

    resnet_test_answers = test_infer(resnet, test)
    efficientnet_test_answers = test_infer(efficientnet, test)
    test_answers = np.concatenate(
        (resnet_test_answers, efficientnet_test_answers), axis=1
    )
    classes = [CLASSES.index(class_name) for class_name in test.classes]

    print(clf.score(test_answers, classes))

    test_answers = clf.predict(test_answers)

    submission = pd.read_csv(test_markup, engine="python")
    submission["category"] = pd.Series(test_answers).apply(lambda x: CLASSES[x])
    submission.to_csv(PROJECT_DIR / "hackaton_ds/submission.csv", index=None)
