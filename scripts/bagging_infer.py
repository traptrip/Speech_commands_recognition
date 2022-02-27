from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from tqdm import tqdm
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from catboost import CatBoostClassifier

from efficientnet_train import CLASSES, DEVICE
from efficientnet_train import TestData
from efficientnet_train import load_model as load_efficientnet_model
from resnet_train import load_model as load_resnet_model

PROJECT_DIR = Path(__file__).resolve().parents[1]
softmax = nn.Softmax(dim=0)
WEIGHTS_PATH = Path("../notebooks/pretrained_models/xvector_finetuned")
DEVICE = "cuda"

enc_classifier = EncoderClassifier.from_hparams(
    source=WEIGHTS_PATH,
    savedir=WEIGHTS_PATH,
    run_opts={"device": DEVICE},
)
audio_normalizer = enc_classifier.audio_normalizer
label_encoder = enc_classifier.hparams.label_encoder
rel_length = torch.tensor([1.0])


def load_audio(path):
    signal, sr = torchaudio.load(str(path), channels_first=False)
    return audio_normalizer(signal, sr)


def resnet_infer(model, test):
    if (PROJECT_DIR / "models/resnet_test_answers.npy").exists():
        with open(PROJECT_DIR / "models/resnet_test_answers.npy", "rb") as f:
            return np.load(f)

    test_answers = list()
    for X, _ in tqdm(test):
        X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        test_answers.append(logists)

    return np.asfarray(test_answers)


def efficient_infer(model, test):
    if (PROJECT_DIR / "models/efficientnet_test_answers.npy").exists():
        with open(PROJECT_DIR / "models/efficientnet_test_answers.npy", "rb") as f:
            return np.load(f)

    test_answers = list()
    for X, _ in tqdm(test):
        X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        test_answers.append(logists)

    return np.asfarray(test_answers)


def xvector_infer_train(enc_classifier):
    if (PROJECT_DIR / "models/xvec_test_ans.npy").exists():
        with open(PROJECT_DIR / "models/xvec_test_ans.npy", "rb") as f:
            return np.load(f)
    test_audio_filepaths = sorted(list((PROJECT_DIR / "data/test").rglob("*.wav")))
    # pred = []
    probas = []
    for audiofile in tqdm(test_audio_filepaths):
        wav = load_audio(audiofile).unsqueeze(0)
        output = enc_classifier.classify_batch(wav, rel_length)
        # class_name = output[-1][-1]
        out_probs = output[0]
        out_probs = softmax(out_probs[0]).cpu().data.numpy()

        true_probs = [
            out_probs[7].item(),
            out_probs[6].item(),
            out_probs[8].item(),
            out_probs[1].item(),
            out_probs[9].item(),
            out_probs[5].item(),
            out_probs[2].item(),
            out_probs[3].item(),
            out_probs[0].item(),
            out_probs[4].item(),
        ]

        probas.append(true_probs)
        # pred.append(class_name)
        # break

    return np.asfarray(probas)


if __name__ == "__main__":
    test_dir = PROJECT_DIR / "data/test"
    test_markup = PROJECT_DIR / "data/sub.csv"
    test = TestData(test_dir, test_markup)

    resnet = load_resnet_model().to(DEVICE)
    resnet.load_state_dict(torch.load(PROJECT_DIR / "models/resnet16_95ep.pt"))
    resnet.eval()

    # efficientnet = load_efficientnet_model().to(DEVICE)
    # efficientnet.load_state_dict(
    #     torch.load(PROJECT_DIR / "models/efficientnet_70ep.pt")
    # )
    # efficientnet.eval()

    # with open(str(PROJECT_DIR / "models/log_reg.pkl"), "rb") as f:
    #     clf = pickle.load(f)
    clf = CatBoostClassifier(
        auto_class_weights="SqrtBalanced",
        # iterations=3000,
        eval_metric="AUC",
    )
    clf.load_model(str(PROJECT_DIR / "models/cat.cbm"))

    resnet_test_answers = resnet_infer(resnet, test)
    with open(PROJECT_DIR / "models/resnet_test_answers.npy", "wb") as f:
        np.save(f, resnet_test_answers)

    # efficientnet_test_answers = efficient_infer(efficientnet, test)
    # with open(PROJECT_DIR / "models/efficientnet_test_answers.npy", "wb") as f:
    #     np.save(f, efficientnet_test_answers)

    xvec_ans = xvector_infer_train(enc_classifier)
    with open(PROJECT_DIR / "models/xvec_test_ans.npy", "wb") as f:
        np.save(f, xvec_ans)

    test_answers = np.concatenate((resnet_test_answers, xvec_ans), axis=1)
    classes = [CLASSES.index(class_name) for class_name in test.classes]

    print(clf.score(test_answers, classes))

    test_answers = clf.predict(test_answers)
    print(test_answers.shape)

    submission = pd.read_csv(test_markup, engine="python")
    submission["category"] = pd.Series(test_answers).apply(lambda x: CLASSES[x])
    submission.to_csv(
        PROJECT_DIR / "submission_BAGGING_resnet_xvector_cat.csv", index=None
    )
