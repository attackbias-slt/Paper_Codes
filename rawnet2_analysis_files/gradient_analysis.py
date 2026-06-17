import os
import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import importlib.util
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
        path = os.path.join(self.base_dir, key + ".flac")

        wav, _ = self.safe_load_audio(path)
        wav = self.pad_crop(wav)

        # TRAIN: (wav, label)
        # DEV/EVAL: (wav, key)
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


Rawnet_2_path = "insert model path"
spec = importlib.util.spec_from_file_location("Rawnet2", Rawnet_2_path)
Rawnet2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Rawnet2)

Model = Rawnet2.RawNet
def build_rawnet2(d_args, ckpt_path, device):
    # RawNet expects a DICT, not a SimpleNamespace
    model = Model(d_args.__dict__, device)

    state = torch.load(
        ckpt_path,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model

def zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

def get_grad_vector(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    return torch.cat(grads).detach()

def grad_for_batch(model, x, y, loss_fn, device):
    model.eval()
    zero_grads(model)

    x = x.to(device)
    y = y.to(device).long()
    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    g_vec = get_grad_vector(model)
    return g_vec, loss.item()


def compute_per_attack_loss(model, loader, device):
    attack_losses = defaultdict(list)
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        for x, y, attacks in loader:
            x = x.to(device)
            y = y.to(device).long()

            logits = model(x)
            loss_sample = F.cross_entropy(logits, y, reduction="none")

            attacks = list(attacks)
            for aid in set(attacks):
                mask = torch.tensor([a == aid for a in attacks], device=device)
                if mask.sum() > 0:
                    attack_losses[aid].append(loss_sample[mask].mean().item())

    return {a: float(np.mean(v)) for a, v in attack_losses.items()}


def compute_per_attack_entropy(model, loader, device):
    attack_entropy = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for x, y, attacks in loader:
            x = x.to(device)
            y = y.to(device).long()
            attacks = list(attacks)

            logits = model(x)
            probs = F.softmax(logits, dim=1).clamp(1e-8, 1-1e-8)
            ent = -(probs * probs.log()).sum(dim=1)

            attacks = list(attacks)
            for aid in set(attacks):
                mask = torch.tensor([a == aid for a in attacks], device=device)
                if mask.sum() > 0:
                    attack_entropy[aid].append(ent[mask].mean().item())

    return {a: float(np.mean(v)) for a, v in attack_entropy.items()}

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    meta_train = "insert metadata training file path"
    base_dir_train = "insert base directory path"

    # Load Rawnet2 config
    import yaml
    from types import SimpleNamespace
    conf_path = "insert model config path"
    with open(conf_path, "r") as f:
        rawnet_conf = yaml.safe_load(f)
    d_args = SimpleNamespace(**rawnet_conf["model"])

    ckpt = "insert model checkpoint path"

    # ---- LOAD METADATA ----
    labels, attack_ids, file_list = parse_asvspoof_meta(meta_train)

    # ---- BUILD ORIGINAL BASE DATASET ----
    train_base = Dataset_ASVspoof2019_train(
        list_IDs=file_list,
        labels=labels,
        base_dir=base_dir_train
    )

    # ---- WRAP DATASET ----
    train_dataset = Wrapped_ASVspoof_Dataset(
        base_dataset=train_base,
        attack_ids=attack_ids,
        nb_samp=d_args.nb_samp)


    # ---- DATALOADER ----
    full_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    model = build_rawnet2(d_args, ckpt, device)

    print("\nComputing per-attack loss…")
    per_attack_loss = compute_per_attack_loss(model, full_loader, device)

    print("\nComputing per-attack entropy…")
    per_attack_entropy = compute_per_attack_entropy(model, full_loader, device)
    
    os.makedirs("metrics_checkpoint", exist_ok=True)
    result = {
        "loss": per_attack_loss,
        "entropy": per_attack_entropy
    }

    with open("metrics_checkpoint/result_configuration_type.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nSaved metrics for checkpoint.")
    print(json.dumps(result, indent=4))
