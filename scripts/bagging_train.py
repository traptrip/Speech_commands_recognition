from pathlib import Path

import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from efficientnet_train import CLASSES, DEVICE
from efficientnet_train import TrainData as EfficientnetTrainData
from efficientnet_train import TestData
from efficientnet_train import load_model as load_efficientnet_model
from resnet_train import TrainData as ResnetTrainData
from resnet_train import TrainData2 as ResnetTrainData2

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
N_NEIGHBORS = 20
classifier = KNeighborsClassifier(
    n_neighbors=N_NEIGHBORS,
    metric="cosine",
    n_jobs=-1,
)


def load_audio(path):
    signal, sr = torchaudio.load(str(path), channels_first=False)
    return audio_normalizer(signal, sr)


def resnet_train_infer1(model):

    noise_dir = PROJECT_DIR / "data/noises"
    train_dir = PROJECT_DIR / "data/train"
    train = ResnetTrainData(train_dir, noise_dir)
    train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0)

    if (PROJECT_DIR / "models/resnet_train_answers.npy").exists():
        with open(PROJECT_DIR / "models/resnet_train_answers.npy", "rb") as f:
            return np.load(f), train.classes

    train_answers = list()
    for X, _ in tqdm(train_loader):
        # X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        train_answers.append(logists)
        # break
    return np.asfarray(train_answers), train.classes


def resnet_train_infer2(model):

    noise_dir = PROJECT_DIR / "data/noises"
    train_dir = PROJECT_DIR / "data/train"
    train = ResnetTrainData2(train_dir, noise_dir)
    train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0)

    if (PROJECT_DIR / "models/resnet2_train_answers.npy").exists():
        with open(PROJECT_DIR / "models/resnet2_train_answers.npy", "rb") as f:
            return np.load(f), train.classes

    train_answers = list()
    for X, _ in tqdm(train_loader):
        # X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        train_answers.append(logists)
        # break
    return np.asfarray(train_answers), train.classes


def efficientnet_train_infer(model):
    if (PROJECT_DIR / "models/efficientnet_train_answers.npy").exists():
        with open(PROJECT_DIR / "models/efficientnet_train_answers.npy", "rb") as f:
            return np.load(f), []

    noise_dir = PROJECT_DIR / "data/noises"
    train_dir = PROJECT_DIR / "data/train"
    train = EfficientnetTrainData(train_dir, noise_dir)
    train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=0)

    train_answers = list()
    for X, _ in tqdm(train_loader):
        # X = X.unsqueeze(0)
        preds = model.forward(X.to(DEVICE))
        logists = softmax(preds[0]).cpu().data.numpy()
        train_answers.append(logists)
        # break

    return np.asfarray(train_answers), train.classes


def xvector_infer_train(enc_classifier):
    if (PROJECT_DIR / "models/xvec_train_ans.npy").exists():
        with open(PROJECT_DIR / "models/xvec_train_ans.npy", "rb") as f:
            return np.load(f), []

    test_audio_filepaths = sorted(list((PROJECT_DIR / "data/train").rglob("*.wav")))
    # pred = []
    probas = []
    classes = []
    embs = []
    for audiofile in tqdm(test_audio_filepaths):
        wav = load_audio(audiofile).unsqueeze(0)

        # emb = enc_classifier.encode_batch(wav, rel_length)
        # embs.append(emb)

        output = enc_classifier.classify_batch(wav, rel_length)
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
        classes.append(audiofile.parts[-2])
        # break

    return np.asfarray(probas), classes


if __name__ == "__main__":
    # test_dir = PROJECT_DIR / "hackaton_ds/test"
    # test_markup = PROJECT_DIR / "hackaton_ds/submission_xvector_cos_sim.csv"
    # test = TestData(test_dir, test_markup)
    # test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0)
    resnet = load_resnet_model().to(DEVICE)
    resnet.load_state_dict(torch.load(PROJECT_DIR / "models/resnet16_95ep.pt"))
    resnet.eval()

    resnet2 = load_resnet_model().to(DEVICE)
    resnet2.load_state_dict(torch.load(PROJECT_DIR / "models/resnet16_2.pt"))
    resnet2.eval()

    efficientnet = load_efficientnet_model().to(DEVICE)
    efficientnet.load_state_dict(
        torch.load(PROJECT_DIR / "models/efficientnet_70ep.pt")
    )
    efficientnet.eval()

    resnet_train_answers, classes1 = resnet_train_infer1(resnet)
    with open(PROJECT_DIR / "models/resnet_train_answers.npy", "wb") as f:
        np.save(f, resnet_train_answers)

    resnet2_train_answers, classes1 = resnet_train_infer2(resnet2)
    with open(PROJECT_DIR / "models/resnet2_train_answers.npy", "wb") as f:
        np.save(f, resnet2_train_answers)

    classes = [CLASSES.index(class_name) for class_name in classes1]
    efficientnet_train_answers, _ = efficientnet_train_infer(efficientnet)
    with open(PROJECT_DIR / "models/efficientnet_train_answers.npy", "wb") as f:
        np.save(f, efficientnet_train_answers)

    xvector_ans, classes2 = xvector_infer_train(enc_classifier)
    with open(PROJECT_DIR / "models/xvec_train_ans.npy", "wb") as f:
        np.save(f, xvector_ans)

    train_answers = np.concatenate(
        (
            resnet_train_answers,
            resnet2_train_answers,
            efficientnet_train_answers,
            xvector_ans,
        ),
        axis=1,
    )

    print(train_answers.shape)

    # clf = LogisticRegression(random_state=0, solver="saga").fit(train_answers, classes)

    # # now you can save it to a file
    # with open(str(PROJECT_DIR / "models/log_reg.pkl"), "wb") as f:
    #     pickle.dump(clf, f)

    clf = CatBoostClassifier(
        task_type="GPU",
        devices=[0],
        auto_class_weights="SqrtBalanced",
        iterations=2500,
        eval_metric="AUC",
    )
    clf.fit(train_answers, classes, verbose=1)
    print("Saving model")
    clf.save_model(str(PROJECT_DIR / "models/cat.cbm"))

    # resnet_test_answers = test_infer(resnet, test)
    # print(resnet_test_answers)
    # efficientnet_test_answers = test_infer(efficientnet, test)
    # print(efficientnet_test_answers)
    # test_answers = np.concatenate(
    #     (resnet_test_answers, efficientnet_test_answers), axis=1
    # )
    # print(test_answers)
