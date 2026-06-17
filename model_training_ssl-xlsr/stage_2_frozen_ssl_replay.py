import argparse
import sys
import os
import itertools
import re
from collections import Counter

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

from data_utils_SSL_19 import (
    genSpoof_list,
    Dataset_ASVspoof2019_train,
    Dataset_ASVspoof2021_eval,
    Dataset_in_the_wild_eval
)
from model import Model
from core_scripts.startup_config import set_random_seed


ATTACK_RE = re.compile(r"A\d{2}")

def parse_stage2_protocol(protocol_path: str):
    entries = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()

            utt_id = toks[1] if len(toks) > 1 else toks[0]
            label_tok = None
            for t in toks:
                tl = t.lower()
                if tl in ("spoof", "bonafide"):
                    label_tok = tl
                    break
            if label_tok is None:
                if toks[-1] in ("0", "1"):
                    label_int = int(toks[-1])
                else:
                    raise ValueError(f"Could not find label (spoof/bonafide or 0/1) in line: {line}")
            else:
                label_int = 0 if label_tok == "spoof" else 1
            attack = None
            for t in toks:
                m = ATTACK_RE.search(t)
                if m:
                    attack = m.group(0)
                    break

            entries.append({"utt_id": utt_id, "label": label_int, "attack": attack, "raw": line})
    return entries

def split_stage2_entries(entries, replay_attacks=None, put_bonafide_in_replay=True):
    if replay_attacks is None:
        replay_attacks = {"A01", "A02", "A05", "A06"}

    replay_ids, new_ids = [], []
    for e in entries:
        if e["label"] == 1 and put_bonafide_in_replay:
            replay_ids.append(e["utt_id"])
        else:
            if e["attack"] in replay_attacks:
                replay_ids.append(e["utt_id"])
            else:
                new_ids.append(e["utt_id"])
    return replay_ids, new_ids


def logits_to_bon_logit(out2: torch.Tensor) -> torch.Tensor:
    return out2[:, 1] - out2[:, 0]  # [B]


@torch.no_grad()
def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_correct = 0
    num_total = 0
    bonafide_correct = 0
    spoof_correct = 0
    bonafide_total = 0
    spoof_total = 0

    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch_x, batch_y in tqdm(dev_loader, desc="Valid", leave=False):
        batch_size = batch_x.size(0)
        num_total += batch_size

        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).long().to(device)

        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)

        # per-class accuracy counts
        for i in range(batch_size):
            label = batch_y[i].item()
            pred = batch_pred[i].item()
            if label == 0:
                spoof_total += 1
                spoof_correct += int(pred == 0)
            else:
                bonafide_total += 1
                bonafide_correct += int(pred == 1)

        num_correct += (batch_pred == batch_y).sum().item()
        batch_loss = criterion(batch_out, batch_y)
        val_loss += batch_loss.item() * batch_size

    val_loss /= max(num_total, 1)
    acc = 100 * num_correct / max(num_total, 1)
    bon_acc = 100 * bonafide_correct / max(bonafide_total, 1)
    spoof_acc = 100 * spoof_correct / max(spoof_total, 1)
    return val_loss, acc, bon_acc, spoof_acc


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    model.eval()

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
    if os.path.exists(save_path):
        os.remove(save_path)

    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader, desc="Scoring", leave=False):
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)

            # score = bonafide logit/prob; keep your old convention if needed
            batch_score = batch_out[:, 1].data.cpu().numpy().ravel()

            with open(save_path, "a+", encoding="utf-8") as fh:
                for f, cm in zip(utt_id, batch_score.tolist()):
                    fh.write(f"{f} {cm}\n")

    print(f"Scores saved to {save_path}", flush=True)


# -----------------------------
# Stage-2 Training (Replay + Constraint)
# -----------------------------
def train_epoch_stage2(
    new_loader,
    replay_loader,
    student,
    teacher,
    optimizer,
    device,
    alpha=1.0,      # new BCE weight
    alpha_r=1.0,    # replay BCE weight
    beta=0.5,       # KD weight
    gamma=5.0,      # constraint weight
    margin=0.0,     # allowed replay loss increase
    max_steps=None  # steps per epoch (None => max(len(new), len(replay)))
):
    """
    Alternates batches from new_loader and replay_loader within the epoch.

    Objective:
      L = alpha * BCE(new)
        + alpha_r * BCE(replay)
        + beta * KL( teacher || student ) on replay
        + gamma * mean( relu( BCE_student(replay) - BCE_teacher(replay) - margin ) )
    """
    student.train()
    teacher.eval()

    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    kl = nn.KLDivLoss(reduction="batchmean")
    eps = 1e-8

    new_iter = itertools.cycle(new_loader)
    rep_iter = itertools.cycle(replay_loader)

    if max_steps is None:
        max_steps = max(len(new_loader), len(replay_loader))

    total_loss_sum = 0.0
    new_bce_sum = 0.0
    rep_bce_sum = 0.0
    kd_sum = 0.0
    constraint_sum = 0.0
    rep_violation_frac_sum = 0.0
    n_steps = 0

    for _ in tqdm(range(max_steps), desc="Train(S2)", leave=False):
        # ---- New batch ----
        x_new, y_new = next(new_iter)
        x_new = x_new.to(device)
        y_new = y_new.view(-1).long().to(device)
        y_new_f = y_new.float()

        out_new = student(x_new)                      # [B,2]
        logit_new = logits_to_bon_logit(out_new)      # [B]
        loss_new_vec = bce_none(logit_new, y_new_f)   # [B]
        loss_new = loss_new_vec.mean()

        # ---- Replay batch ----
        x_rep, y_rep = next(rep_iter)
        x_rep = x_rep.to(device)
        y_rep = y_rep.view(-1).long().to(device)
        y_rep_f = y_rep.float()

        out_s = student(x_rep)
        logit_s = logits_to_bon_logit(out_s)
        loss_s_vec = bce_none(logit_s, y_rep_f)       # [B]
        loss_bce_rep = loss_s_vec.mean()

        with torch.no_grad():
            out_t = teacher(x_rep)
            loss_t_vec = bce_none(logits_to_bon_logit(out_t), y_rep_f)  # [B]

        # Constraint: penalize only if student worse than teacher on replay samples
        constraint_vec = F.relu(loss_s_vec - loss_t_vec - margin)  # [B]
        loss_constraint = constraint_vec.mean()

        # KD on replay (teacher distribution -> student)
        logp_s = F.log_softmax(out_s, dim=1)
        p_t = F.softmax(out_t, dim=1).clamp(min=eps)
        loss_kd = kl(logp_s, p_t)

        loss_total = alpha * loss_new + alpha_r * loss_bce_rep + beta * loss_kd + gamma * loss_constraint

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        total_loss_sum += loss_total.item()
        new_bce_sum += loss_new.item()
        rep_bce_sum += loss_bce_rep.item()
        kd_sum += loss_kd.item()
        constraint_sum += loss_constraint.item()

        violations = (loss_s_vec > (loss_t_vec + margin)).float().mean().item()
        rep_violation_frac_sum += violations

        n_steps += 1

    return {
        "loss_total": total_loss_sum / max(n_steps, 1),
        "loss_new_bce": new_bce_sum / max(n_steps, 1),
        "loss_rep_bce": rep_bce_sum / max(n_steps, 1),
        "loss_kd": kd_sum / max(n_steps, 1),
        "loss_constraint": constraint_sum / max(n_steps, 1),
        "rep_violation_frac": rep_violation_frac_sum / max(n_steps, 1),
    }


def freeze_model(m: nn.Module):
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-2 Continual Replay Training (Split replay from stage2 protocol)")

    # Paths (kept for compatibility with your eval modes)
    parser.add_argument("--database_path", type=str, default="/path/to/your/database/")
    parser.add_argument("--protocols_path", type=str, default="/path/to/your/database/")
    parser.add_argument("--comment", type=str, default="stage2")

    # Determinism toggles expected by startup_config.py
    parser.add_argument("--cudnn_deterministic_toggle", action="store_true",
                        help="Enable deterministic CuDNN (may slow training).")
    parser.add_argument("--cudnn_benchmark_toggle", action="store_true",
                        help="Enable CuDNN benchmark (non-deterministic, faster).")

    # Stage-1 best checkpoint (teacher)
    parser.add_argument("--stage1_best_ckpt", type=str, required=True, help="Path to best stage-1 .pth checkpoint")

    # Stage-2 protocol (single source for NEW + REPLAY)
    parser.add_argument("--stage2_protocol", type=str, required=True,
                        help="Protocol txt for stage-2 data (contains A01..A06 etc.)")
    parser.add_argument("--stage2_base_dir", type=str, required=True,
                        help="Base dir for stage-2 audio (train flac folder)")

    # Validation
    parser.add_argument("--dev_protocol", type=str, required=True, help="Protocol txt for validation (dev)")
    parser.add_argument("--dev_base_dir", type=str, required=True, help="Base dir for validation audio (dev flac folder)")

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_workers", type=int, default=8)

    # Loss weights
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--alpha_r", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--steps_per_epoch", type=int, default=0, help="0 => auto, else fixed steps per epoch")

    # Misc
    parser.add_argument("--track", type=str, default="DF", choices=["LA", "In-the-Wild", "DF"])
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_output", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Optional init checkpoint for student")

    # Replay split controls
    parser.add_argument("--replay_attacks", type=str, default="A01,A02,A03,A05,A06",
                        help="Comma-separated attack IDs to treat as replay.")
    parser.add_argument("--bonafide_in_replay", action="store_true", default=True,
                        help="Include bonafide trials in replay set (recommended).")

    ##===================================================Rawboost data augmentation ======================================================================#
    parser.add_argument('--algo', type=int, default=3,
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), '
                             '5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

    # LnL_convolutive_noise parameters
    parser.add_argument('--nBands', type=int, default=5,
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20,
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000,
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100,
                        help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000,
                        help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10,
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100,
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0,
                        help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0,
                        help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5,
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20,
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5,
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10,
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2,
                        help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10,
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40,
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')
    ##===================================================Rawboost data augmentation ======================================================================#

    args = parser.parse_args()


    set_random_seed(args.seed, args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    teacher = Model(args, device)
    teacher = nn.DataParallel(teacher).to(device)
    teacher.load_state_dict(torch.load(args.stage1_best_ckpt, map_location=device))
    freeze_model(teacher)
    print(f"[Stage2] Teacher loaded and frozen: {args.stage1_best_ckpt}", flush=True)

    student = Model(args, device)
    student = nn.DataParallel(student).to(device)

    if args.model_path:
        student.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"[Stage2] Student initialized from: {args.model_path}", flush=True)
    else:
        student.load_state_dict(torch.load(args.stage1_best_ckpt, map_location=device))
        print("[Stage2] Student initialized from teacher (best stage-1).", flush=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------
    # Optional EVAL modes (unchanged)
    # -----------------------------
    if args.track == "In-the-Wild":
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        eval_set = Dataset_in_the_wild_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, student, device, args.eval_output)
        sys.exit(0)

    if args.eval:
        file_eval = genSpoof_list(dir_meta=os.path.join(args.protocols_path), is_train=False, is_eval=True)
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path))
        produce_evaluation_file(eval_set, student, device, args.eval_output)
        sys.exit(0)

    # -----------------------------
    # Datasets: NEW (stage2) + REPLAY (from same stage2_protocol) + DEV
    # -----------------------------
    replay_attacks = {a.strip() for a in args.replay_attacks.split(",") if a.strip()}
    entries = parse_stage2_protocol(args.stage2_protocol)
    replay_ids, new_ids = split_stage2_entries(
        entries,
        replay_attacks=replay_attacks,
        put_bonafide_in_replay=args.bonafide_in_replay
    )

    print(f"[Stage2] # new trials: {len(new_ids)}", flush=True)
    print(f"[Stage2] # replay trials: {len(replay_ids)}", flush=True)

    # Robust distributions: use parsed entries instead of utt_id substring matching
    utt2attack = {e["utt_id"]: (e["attack"] if e["attack"] is not None else "NONE") for e in entries}
    print("[Stage2] Replay attack distribution:", Counter(utt2attack.get(u, "MISSING") for u in replay_ids), flush=True)
    print("[Stage2] New attack distribution:", Counter(utt2attack.get(u, "MISSING") for u in new_ids), flush=True)

    # Load labels once from stage2 protocol (genSpoof_list produces a label dict keyed by utt_id)
    d_label_all, _ = genSpoof_list(dir_meta=args.stage2_protocol, is_train=True, is_eval=False)

    # NEW dataset
    stage2_set = Dataset_ASVspoof2019_train(
        args, list_IDs=new_ids, labels=d_label_all, base_dir=args.stage2_base_dir, algo=args.algo
    )
    stage2_loader = DataLoader(
        stage2_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    # REPLAY dataset (same base dir, filtered ids)
    replay_set = Dataset_ASVspoof2019_train(
        args, list_IDs=replay_ids, labels=d_label_all, base_dir=args.stage2_base_dir, algo=args.algo
    )
    replay_loader = DataLoader(
        replay_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
    )

    # DEV (validation)
    d_label_dev, file_dev = genSpoof_list(dir_meta=args.dev_protocol, is_train=False, is_eval=False)
    print(f"[Stage2] # dev trials: {len(file_dev)}", flush=True)
    dev_set = Dataset_ASVspoof2019_train(
        args, list_IDs=file_dev, labels=d_label_dev, base_dir=args.dev_base_dir, algo=args.algo
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False
    )

    # -----------------------------
    # Logging + Saving
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    model_tag = f"stage2_replay_from_s2proto_{args.track}_bs{args.batch_size}_lr{args.lr}_{args.comment}"
    model_save_path = os.path.join("models", model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    writer = SummaryWriter(f"logs/{model_tag}")

    # -----------------------------
    # Training loop (Stage 2)
    # -----------------------------
    for epoch in range(args.num_epochs):
        max_steps = None if args.steps_per_epoch == 0 else args.steps_per_epoch

        stats = train_epoch_stage2(
            new_loader=stage2_loader,
            replay_loader=replay_loader,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            device=device,
            alpha=args.alpha,
            alpha_r=args.alpha_r,
            beta=args.beta,
            gamma=args.gamma,
            margin=args.margin,
            max_steps=max_steps
        )

        val_loss, acc, bon_acc, spoof_acc = evaluate_accuracy(dev_loader, student, device)

        # Tensorboard
        writer.add_scalar("train/loss_total", stats["loss_total"], epoch)
        writer.add_scalar("train/loss_new_bce", stats["loss_new_bce"], epoch)
        writer.add_scalar("train/loss_rep_bce", stats["loss_rep_bce"], epoch)
        writer.add_scalar("train/loss_kd", stats["loss_kd"], epoch)
        writer.add_scalar("train/loss_constraint", stats["loss_constraint"], epoch)
        writer.add_scalar("train/replay_violation_frac", stats["rep_violation_frac"], epoch)

        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/acc", acc, epoch)
        writer.add_scalar("val/acc_bonafide", bon_acc, epoch)
        writer.add_scalar("val/acc_spoof", spoof_acc, epoch)

        print(
            f"\n[Epoch {epoch}] "
            f"TrainTot={stats['loss_total']:.4f} | NewBCE={stats['loss_new_bce']:.4f} | "
            f"RepBCE={stats['loss_rep_bce']:.4f} | KD={stats['loss_kd']:.4f} | "
            f"Constr={stats['loss_constraint']:.4f} | RepViol%={100*stats['rep_violation_frac']:.2f}% || "
            f"ValLoss={val_loss:.4f} | Acc={acc:.2f}% | BonAcc={bon_acc:.2f}% | SpoofAcc={spoof_acc:.2f}%",
            flush=True
        )

        ckpt_path = os.path.join(model_save_path, f"epoch_{epoch}.pth")
        torch.save(student.state_dict(), ckpt_path)

    writer.close()
    print("[Stage2] Done.", flush=True)
