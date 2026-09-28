import argparse
import ast
import hashlib
import math
import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mtg_shared_split import (
    DEFAULT_OUTPUT, DEFAULT_SPLIT, load_shared_split, save_benchmark_result,
    score_probabilities,
)


# -----------------------------
# Engineered regex indicators (feature engineering)
# -----------------------------
EFFECT_PATTERNS: List[Tuple[str, str]] = [
    # interaction
    ("destroy", r"\bdestroy\b"),
    ("exile", r"\bexile\b"),
    ("counter_target", r"\bcounter target\b"),
    ("sacrifice", r"\bsacrifice\b"),
    ("discard", r"\bdiscard\b"),
    ("return_to_hand", r"\breturn\b.*\bto (its|their) owner's hand\b"),
    ("tap_target", r"\btap target\b"),
    ("doesnt_untap", r"\bdoesn'?t untap\b"),
    ("damage", r"\bdeals?\b.*\bdamage\b"),

    # advantage / selection
    ("draw", r"\bdraw\b"),
    ("scry", r"\bscry\b"),
    ("surveil", r"\bsurveil\b"),
    ("investigate", r"\binvestigate\b"),
    ("look_at_top", r"\blook at the top\b"),

    # ramp / mana / tutor-ish
    ("add_mana", r"\badd\s+\{[^\}]+\}"),
    ("search_library", r"\bsearch your library\b"),
    ("search_land", r"\bsearch your library\b.*\bland\b"),
    ("onto_battlefield", r"\bonto the battlefield\b"),
    ("create_treasure", r"\bcreate\b.*\btreasure\b"),

    # board-wide / scaling / flexibility
    ("each", r"\beach\b"),
    ("all", r"\ball\b"),
    ("choose_one", r"\bchoose one\b"),
    ("choose_two", r"\bchoose two\b"),
    ("up_to", r"\bup to\b"),
    ("for_each", r"\bfor each\b"),
    ("equal_to", r"\bequal to\b"),
    ("has_x", r"\bwhere x is\b|\bx\b"),

    # tokens / subtype cues
    ("create_token", r"\bcreate\b.*\btoken\b"),
    ("equip", r"\bequip\b"),
    ("enchant", r"\benchant\b"),
    ("crew", r"\bcrew\b"),
]
# -----------------------------
# Feature engineering helpers
# -----------------------------
@dataclass
class FeatureConfig:
    hash_dim: int = 512
    use_text_indicators: bool = True


@dataclass
class Normalizer:
    mean: List[float]
    std: List[float]


def stable_hash_to_index(token: str, dim: int) -> int:
    h = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(h, 16) % dim


def safe_literal_list(x) -> List[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None]
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = ast.literal_eval(s)
                if isinstance(v, list):
                    return [str(i) for i in v if i is not None]
            except Exception:
                return []
        return [s] if s else []
    return []


def to_float_or_nan(x) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except Exception:
        # e.g. power="*"
        return float("nan")


def normalize_type_line(type_line: str) -> str:
    # Fix common mojibake dash issues seen in MTG exports
    return (
        (type_line or "")
        .replace("â€”", "—")
        .replace("â€“", "—")
        .replace("—", " — ")
        .strip()
    )


def extract_type_tokens(type_line: str) -> Tuple[List[str], List[str]]:
    tl = normalize_type_line(type_line)
    if "—" in tl:
        left, right = [p.strip() for p in tl.split("—", 1)]
    elif "-" in tl:
        left, right = [p.strip() for p in tl.split("-", 1)]
    else:
        left, right = tl.strip(), ""

    left_tokens = [t.lower() for t in re.split(r"\s+", left) if t]
    right_tokens = [t.lower() for t in re.split(r"\s+", right) if t]
    return left_tokens, right_tokens


def fit_normalizer(x_train: np.ndarray) -> Normalizer:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return Normalizer(mean=mean.astype(np.float32).tolist(), std=std.astype(np.float32).tolist())


def apply_normalizer(x: np.ndarray, norm: Normalizer) -> np.ndarray:
    mean = np.array(norm.mean, dtype=np.float32)
    std = np.array(norm.std, dtype=np.float32)
    return (x - mean) / std


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (counts + 1e-9)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def featurize_row(row: pd.Series, cfg: FeatureConfig, compiled_patterns) -> Tuple[np.ndarray, np.ndarray]:
    """
    Outputs:
      x_hash: (hash_dim,)
      x_num:  engineered numeric vector incl. regex counts
    """
    x_hash = np.zeros((cfg.hash_dim,), dtype=np.float32)

    # Categorical: type/subtype/supertype/rarity/layout/colors => hashed bag
    type_line = str(row.get("type", "") or "")
    types, subtypes_from_type = extract_type_tokens(type_line)

    for t in types:
        x_hash[stable_hash_to_index(f"type:{t}", cfg.hash_dim)] += 1.0
    for st in subtypes_from_type:
        x_hash[stable_hash_to_index(f"subtype:{st}", cfg.hash_dim)] += 1.0

    for st in [s.lower() for s in safe_literal_list(row.get("subtypes", None))]:
        if st:
            x_hash[stable_hash_to_index(f"subtype:{st}", cfg.hash_dim)] += 1.0
    for su in [s.lower() for s in safe_literal_list(row.get("supertypes", None))]:
        if su:
            x_hash[stable_hash_to_index(f"super:{su}", cfg.hash_dim)] += 1.0

    rarity = str(row.get("rarity", "") or "").strip().lower()
    if rarity:
        x_hash[stable_hash_to_index(f"rarity:{rarity}", cfg.hash_dim)] += 1.0

    layout = str(row.get("layout", "") or "").strip().lower()
    if layout:
        x_hash[stable_hash_to_index(f"layout:{layout}", cfg.hash_dim)] += 1.0

    colors = safe_literal_list(row.get("color_identity", None))
    if not colors:
        colors = safe_literal_list(row.get("colors", None))
    colors = [c.strip().upper() for c in colors if c]
    for c in colors:
        x_hash[stable_hash_to_index(f"color:{c}", cfg.hash_dim)] += 1.0
    if len(colors) == 0:
        x_hash[stable_hash_to_index("color:COLORLESS", cfg.hash_dim)] += 1.0

    # Numeric + missingness
    power = to_float_or_nan(row.get("power", None))
    toughness = to_float_or_nan(row.get("toughness", None))
    loyalty = to_float_or_nan(row.get("loyalty", None))

    p_m = float(math.isnan(power))
    t_m = float(math.isnan(toughness))
    l_m = float(math.isnan(loyalty))

    power = 0.0 if math.isnan(power) else power
    toughness = 0.0 if math.isnan(toughness) else toughness
    loyalty = 0.0 if math.isnan(loyalty) else loyalty

    is_creature = 1.0 if "creature" in types else 0.0
    is_planeswalker = 1.0 if "planeswalker" in types else 0.0
    num_colors = float(len(colors))

    # Engineered text indicators via regex counts
    text = str(row.get("text", "") or "")
    text_lc = text.lower()

    word_count = float(len(re.findall(r"\w+", text_lc)))
    line_count = float(text_lc.count("\n") + (1 if text_lc else 0))
    digit_count = float(len(re.findall(r"\d", text_lc)))
    brace_count = float(text_lc.count("{"))
    pw_ability_lines = float(len(re.findall(r"(?m)^[+\-0]", text_lc))) if is_planeswalker else 0.0

    indicators = []
    if cfg.use_text_indicators:
        for _, pat in compiled_patterns:
            indicators.append(float(len(pat.findall(text_lc))))
    else:
        indicators = [0.0] * len(compiled_patterns)

    x_num = np.array(
        [
            is_creature,
            is_planeswalker,
            num_colors,
            power, toughness, loyalty,
            p_m, t_m, l_m,
            word_count, line_count, digit_count, brace_count, pw_ability_lines,
            *indicators
        ],
        dtype=np.float32
    )

    return x_hash, x_num


class CMCSoftmaxNet(nn.Module):
    """
    Smaller architecture to prevent overfitting with GroupShuffleSplit.
    512 → 256 → 128 → n_classes
    """
    def __init__(self, in_dim: int, n_classes: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),  
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.35),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def expected_value_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
    return (probs * classes[None, :]).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, default="all_mtg_cards (2).xlsx", help="Path to MTG cards Excel file")
    ap.add_argument("--sheet", type=str, default=None, help="default: first sheet")
    ap.add_argument("--hash-dim", type=int, default=512)  # Reduced to prevent overfitting
    ap.add_argument("--epochs", type=int, default=150,
                    help="Maximum epochs; at least 150 are required")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-mode", type=str, default="drop", choices=["cap", "drop"],
                    help="cap: cmc>10 mapped to 10 (10+ bucket); drop: remove cmc>10 rows")
    ap.add_argument("--out-prefix", type=str, default="mtg_cmc_softmax_0_10")
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    if args.epochs < 150:
        ap.error("--epochs must be at least 150 for the compact MLP")
    print(f"Compact MLP epoch limit: {args.epochs} (minimum 150)", flush=True)

    if args.cap_mode != "drop" or args.sheet is not None:
        raise ValueError("Universal benchmark requires --cap-mode drop and the first sheet")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    split = load_shared_split(Path(args.xlsx), args.split_file, args.seed)
    df = split.cards
    y = split.target
    n_classes = 11

    # Regex compilation
    compiled_patterns = [(name, re.compile(pat)) for name, pat in EFFECT_PATTERNS]
    cfg = FeatureConfig(hash_dim=args.hash_dim, use_text_indicators=True)

    # Build feature matrices
    x_hash = np.zeros((len(df), cfg.hash_dim), dtype=np.float32)
    x_num_list = []
    for i in range(len(df)):
        h, n = featurize_row(df.iloc[i], cfg, compiled_patterns)
        x_hash[i] = h
        x_num_list.append(n)
    x_num = np.stack(x_num_list, axis=0).astype(np.float32)

    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx

    # Normalize numeric based on train split
    norm = fit_normalizer(x_num[train_idx])
    x_num_n = apply_normalizer(x_num, norm)

    x_all = np.concatenate([x_hash, x_num_n], axis=1).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Samples: {len(df)} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print("cap_mode:", args.cap_mode)

    def make_loader(indices, shuffle):
        xt = torch.tensor(x_all[indices], dtype=torch.float32)
        yt = torch.tensor(y[indices], dtype=torch.long)
        ds = torch.utils.data.TensorDataset(xt, yt)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    model = CMCSoftmaxNet(in_dim=x_all.shape[1], n_classes=n_classes).to(device)

    class_w = compute_class_weights(y[train_idx], n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_acc = -1.0
    best_state = None
    patience = 20
    patience_left = patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(len(train_idx), 1)

        model.eval()
        val_loss = 0.0
        correct = 0
        n = 0
        mae_ev = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                ev = expected_value_from_logits(logits)
                mae_ev += torch.abs(ev - yb.float()).sum().item()
                n += xb.size(0)

        val_loss /= max(len(val_idx), 1)
        val_acc = correct / max(n, 1)
        val_mae = mae_ev / max(n, 1)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f} | val_MAE(E[cmc])={val_mae:.3f}")
        
        # Step the scheduler
        scheduler.step(val_acc)

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0 and epoch >= 150:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Training completed after {epoch} epochs", flush=True)

    model.eval()
    test_probabilities = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            test_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    test_metrics, test_predicted = score_probabilities(
        np.concatenate(test_probabilities), y[test_idx], split.groups[test_idx])
    test_metrics.update({"feature_count": int(x_all.shape[1]),
                         "checkpoint_selection": "highest validation exact accuracy",
                         "trained_epochs": epoch})
    print(f"Test | exact={test_metrics['exact_accuracy']:.3%} | "
          f"within 1={test_metrics['within_one_accuracy']:.3%} | "
          f"class MAE={test_metrics['class_mae']:.3f}")
    save_benchmark_result(split, args.output_dir, "compact_mlp", test_metrics, test_predicted)

    # Save
    model_path = f"{args.out_prefix}_model.pt"
    meta_path = f"{args.out_prefix}_meta.json"

    torch.save(model.state_dict(), model_path)
    meta = {
        "hash_dim": cfg.hash_dim,
        "cap_mode": args.cap_mode,
        "n_classes": n_classes,
        "patterns": [n for n, _ in EFFECT_PATTERNS],
        "normalizer": asdict(norm),
        "best_val_acc": float(best_val_acc),
        "trained_epochs": epoch,
        "test_acc": test_metrics["exact_accuracy"],
        "test_mae_expected": test_metrics["expected_value_mae"],
        "test_metrics": test_metrics,
        "split_sha256": split.split_sha256,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", model_path)
    print("Saved:", meta_path)


if __name__ == "__main__":
    main()
