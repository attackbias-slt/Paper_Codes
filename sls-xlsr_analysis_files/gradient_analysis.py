import os
import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import importlib.util
from types import SimpleNamespace
import librosa
from data_utils import Dataset_ASVspoof2019_train
def parse_asvspoof_meta(meta_path):
    """
    Metadata lines look like:
    LA_0079 LA_T_1029621 - A05 spoof
    LA_0079 LA_T_1023001 - bonafide bonafide
    """
    labels = {}
    attack_ids = {}
    file_list = []

    with open(meta_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            key = parts[1]
            attack = parts[2].lower()
            decision = parts[3].lower()

            if decision == "bonafide":
                labels[key] = 0
                attack_ids[key] = "bonafide"
            else:
                labels[key] = 1
                attack_ids[key] = attack.upper()

            file_list.append(key)

    return labels, attack_ids, file_list


class Wrapped_ASVspoof_Dataset(torch.utils.data.Dataset):
    """
    Wraps ASVspoof dataset to:
    - load audio safely
    - pad/crop to fixed length
    - return (wav, label, attack_id)
    """

    def __init__(self, base_dataset, attack_ids, nb_samp=64600):
        self.base = base_dataset
        self.attack_ids = attack_ids
        self.nb_samp = nb_samp

        self.list_IDs = base_dataset.list_IDs
        self.labels = getattr(base_dataset, "labels", None)
        self.base_dir = base_dataset.base_dir

    def safe_load_audio(self, path):
        try:
            wav, sr = torchaudio.load(path)
            wav = wav[0]
        except Exception as e:
            print(f"[WARNING] Failed to load {path}. Using silence. {e}")
            wav = torch.zeros(self.nb_samp)
        return wav, 16000

    def pad_crop(self, wav):
        if wav.shape[0] < self.nb_samp:
            wav = F.pad(wav, (0, self.nb_samp - wav.shape[0]))
        else:
            wav = wav[:self.nb_samp]
        return wav

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        key = self.list_IDs[idx]

        # Correct ASVspoof LA path format
        path = os.path.join(self.base_dir, key + ".flac")

        wav, _ = self.safe_load_audio(path)
        wav = self.pad_crop(wav)

        if len(item) == 2:
            _, y = item
            try:
                label = int(y)
            except:
                label = 1 if self.attack_ids[key] != "bonafide" else 0
        else:
            label = None

        attack_id = self.attack_ids[key]

        return wav, label, attack_id

SSL_MODEL_PATH = "insert path to ssl model file"

spec = importlib.util.spec_from_file_location("ssl_model", SSL_MODEL_PATH)
ssl_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssl_model)
Model = ssl_model.Model


def build_ssl_model(ckpt_path, device):
    model = Model(args=None, device=device)

    state = torch.load(ckpt_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]

    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Freeze SSL backbone
    for p in model.ssl_model.model.parameters():
        p.requires_grad = False

    return model

def compute_per_attack_loss(model, loader, device):
    loss_fn = torch.nn.NLLLoss(reduction="none")
    attack_losses = defaultdict(list)

    with torch.no_grad():
        for x, y, attacks in loader:
            x = x.to(device).unsqueeze(-1)   # [B, T, 1]
            y = y.to(device)

            log_probs = model(x)
            loss = loss_fn(log_probs, y)

            attacks = np.array(attacks)
            for aid in np.unique(attacks):
                mask = torch.from_numpy(attacks == aid).to(device)
                attack_losses[aid].append(loss[mask].mean().item())

    return {k: float(np.mean(v)) for k, v in attack_losses.items()}


def compute_per_attack_entropy(model, loader, device):
    attack_entropy = defaultdict(list)

    with torch.no_grad():
        for x, _, attacks in loader:
            x = x.to(device).unsqueeze(-1)

            log_probs = model(x)
            probs = log_probs.exp()
            entropy = -(probs * log_probs).sum(dim=1)

            attacks = np.array(attacks)
            for aid in np.unique(attacks):
                mask = torch.from_numpy(attacks == aid).to(device)
                attack_entropy[aid].append(entropy[mask].mean().item())

    return {k: float(np.mean(v)) for k, v in attack_entropy.items()}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    meta_train = "insert metadata training file path"
    base_dir_train = "insert file path for base directory"
    ckpt = "insert checkpoint path"

    labels, attack_ids, file_list = parse_asvspoof_meta(meta_train)

    train_base = Dataset_ASVspoof2019_train(
        list_IDs=file_list,
        labels=labels,
        base_dir=base_dir_train
    )

    dataset = WrappedDataset(base_dataset, attack_ids)

    loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = build_ssl_model(ckpt, device)

    print("Computing per-attack loss...")
    loss_res = compute_per_attack_loss(model, loader, device)

    print("Computing per-attack entropy...")
    entropy_res = compute_per_attack_entropy(model, loader, device)

    os.makedirs("metrics_checkpoint", exist_ok=True)
    out = {"loss": loss_res, "entropy": entropy_res}

    with open("metrics_checkpoint/results_configuration_type.json", "w") as f:
        json.dump(out, f, indent=4)

    print("Saved metrics:")
    print(json.dumps(out, indent=4))
