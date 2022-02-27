import time
from enum import auto
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as opt
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MAX_AUDIO_LEN = 16000  # в отсчётах sr
DEVICE = "cuda:0"
BATCH_SIZE = 64
N_EPOCHS = 10000
VAL_ITER = 5
CLASSES = [
    "down",
    "go",
    "left",
    "no",
    "off",
    "on",
    "right",
    "stop",
    "up",
    "yes",
]


class MelCreator:
    def __init__(self) -> None:
        self.make_melspec = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=128,
            f_min=55.0,
            f_max=7600,
            pad=0,
            n_mels=128,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=True,
            center=False,
            pad_mode="reflect",
            onesided=True,
            norm="slaney",  #'slaney',
            mel_scale="htk",
        )
        self.atdb = T.AmplitudeToDB(stype="power", top_db=80)
        # self.atdb = T.AmplitudeToDB(stype="power", top_db=100)

    def __call__(self, audio):
        melspec = self.atdb(self.make_melspec(audio))
        return melspec


class TrainData(Dataset):
    def __init__(self, audio_dir: Path, noise_dir: Path) -> None:
        super().__init__()
        self.audio_len = MAX_AUDIO_LEN
        self.mel_creator = MelCreator()

        self.audio_paths = list()
        self.classes = list()
        audioss = sorted(list(audio_dir.rglob("*.wav")))
        for wav_path in audioss:
            audio, _ = torchaudio.load(wav_path)
            self.audio_paths.append(audio)
            self.classes.append(wav_path.parts[-2])

        self.noises_paths = list()
        for wav_path in noise_dir.iterdir():
            noise, _ = torchaudio.load(wav_path)
            self.noises_paths.append(noise)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        audio = self.audio_paths[idx]
        audio = self.__pad_audio(audio)

        noise = self.__get_random_noize()
        audio = self.__add_noise(audio, noise, 0, 6)
        # if np.random.randint(0, 1):
        #     noise = self.__get_random_noize()
        #     audio = self.__add_noise(audio, noise, 0, 6)

        audio = audio / audio.abs().max()
        melspec = self.mel_creator(audio)
        # melspec = melspec * np.random.uniform(0.75, 1.5)

        return melspec, CLASSES.index(self.classes[idx])

    def __get_random_noize(self):
        noise_idx = np.random.randint(0, len(self.noises_paths) - 1)
        noise = self.noises_paths[noise_idx]
        if noise.shape[0] == 2:
            noise = noise[np.random.randint(0, 2)]
            noise = noise.unsqueeze(0)

        return noise

    def __pad_audio(self, audio):
        if self.audio_len - audio.shape[-1] > 0:
            i = np.random.randint(0, self.audio_len - audio.shape[-1])
        else:
            i = 0
        pad_patern = (i, self.audio_len - audio.shape[-1] - i)
        audio = F.pad(audio, pad_patern, "constant").detach()
        return audio

    def __add_noise(self, clean, noise, min_amp, max_amp):
        noise_amp = np.random.uniform(min_amp, max_amp)
        # так как шумная запись длиннее, то выбираем случайный момент начала шумной записи
        start = np.random.randint(0, noise.shape[1] - clean.shape[1] + 1)
        noise_part = noise[:, start : start + clean.shape[1]]

        if noise_part.abs().max() == 0:
            return clean

        # накладываем шум
        noise_mult = clean.abs().max() / noise_part.abs().max() * noise_amp
        return (clean + noise_part * noise_mult) / (1 + noise_amp)


class TestData(Dataset):
    def __init__(self, audio_dir: Path, markup_path) -> None:
        super().__init__()
        self.audio_len = MAX_AUDIO_LEN
        self.mel_creator = MelCreator()

        self.audio_paths = list()
        self.classes = list()
        markup = pd.read_csv(markup_path)
        for file_name, category in markup.values:
            audio, _ = torchaudio.load(audio_dir / f"{file_name}.wav")
            self.audio_paths.append(audio)
            self.classes.append(category)

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):
        audio = self.audio_paths[idx]
        audio = self.__pad_audio(audio)
        audio = audio / audio.abs().max()
        melspec = self.mel_creator(audio)

        return melspec, CLASSES.index(self.classes[idx])

    def __pad_audio(self, audio):
        if self.audio_len - audio.shape[-1] > 0:
            i = np.random.randint(0, self.audio_len - audio.shape[-1])
        else:
            i = 0
        pad_patern = (i, self.audio_len - audio.shape[-1] - i)
        audio = F.pad(audio, pad_patern, "constant").detach()

        return audio


def load_model():
    # # === Efficientnet ===
    # model = torch.hub.load(
    #     "NVIDIA/DeepLearningExamples:torchhub",
    #     "nvidia_efficientnet_b0",
    #     pretrained=False,
    # )
    # model.stem.conv = nn.Conv2d(
    #     1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
    # )
    # num_ftrs = model.classifier.fc.in_features
    # model.classifier.fc = nn.Linear(num_ftrs, 10, bias=True)

    # === Resnet 16 ===
    model = models.resnet18()
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10, bias=True)

    return model


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    noise_dir = Path("hackaton_ds/noises")
    train_dir = Path("hackaton_ds/train")
    train = TrainData(train_dir, noise_dir)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    test_dir = Path("hackaton_ds/test")
    test_markup = Path("hackaton_ds/submission_xvector_cos_sim.csv")
    test = TestData(test_dir, test_markup)
    val_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = load_model().to(DEVICE)
    optimizer = opt.AdamW(model.parameters(), lr=0.001)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    for epoch in range(N_EPOCHS):
        start = time.time()
        model.train()

        train_acc = list()
        train_loss = list()
        for X, Y in tqdm(train_loader):
            preds = model.forward(X.to(DEVICE))
            loss = F.cross_entropy(
                preds,
                Y.to(DEVICE),
            )
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().data.numpy())
            train_acc.extend((torch.argmax(preds, dim=-1).cpu() == Y).data.numpy())

        if epoch % VAL_ITER == 0:
            model.eval()

            val_acc = list()
            val_loss = list()
            for X, Y in tqdm(val_loader):
                preds = model.forward(X.to(DEVICE))
                loss = F.cross_entropy(
                    preds,
                    Y.to(DEVICE),
                )

                val_loss.append(loss.cpu().data.numpy())
                val_acc.extend((torch.argmax(preds, dim=-1).cpu() == Y).data.numpy())
                torch.save(model.state_dict(), f"models/efficientnet_{epoch}ep.pt")

        epoch_time = time.time() - start
        train_acc = sum(train_acc) / len(train_acc)
        train_loss = sum(train_loss) / len(train_loss)
        val_acc1 = sum(val_acc) / len(val_acc)
        val_loss1 = sum(val_loss) / len(val_loss)
        print(
            f"Epoch time: {epoch_time:.2}|Train acc: {train_acc:.2}|Train loss: {train_loss:.2}|Val acc: {val_acc1:.2}|Val loss: {val_loss1:.2}"
        )
