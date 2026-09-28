import argparse
import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import openpyxl
import torch
import torch.nn as nn
from mtg_shared_split import (
    DEFAULT_OUTPUT, DEFAULT_SPLIT, load_shared_split, save_benchmark_result,
    score_probabilities,
)


# -----------------------------
# Feature engineering (no NLP)
# -----------------------------

# Regex indicator counters from rules text (still "feature engineering").
EFFECT_PATTERNS: List[Tuple[str, str]] = [
    # interaction / removal
    ("destroy", r"\bdestroy\b"),
    ("exile", r"\bexile\b"),
    ("counter_target", r"\bcounter target\b"),
    ("sacrifice", r"\bsacrifice\b"),
    ("discard", r"\bdiscard\b"),
    ("return_to_hand", r"\breturn\b.*\bto (its|their) owner's hand\b"),
    ("tap_target", r"\btap target\b"),
    ("doesnt_untap", r"\bdoesn'?t untap\b"),
    ("damage", r"\bdeals?\b.*\bdamage\b"),
    ("fight", r"\bfight\b"),
    ("mill", r"\bmill\b"),
    
    # card advantage / selection
    ("draw", r"\bdraw\b"),
    ("scry", r"\bscry\b"),
    ("surveil", r"\bsurveil\b"),
    ("investigate", r"\binvestigate\b"),
    ("look_at_top", r"\blook at the top\b"),
    ("reveal", r"\breveal\b"),
    
    # ramp / mana
    ("add_mana_symbol", r"\badd\s+\{[wubrgc0-9x]+\}"),
    ("search_land", r"\bsearch your library\b.*\bland\b"),
    ("create_treasure", r"\bcreate\b.*\btreasure\b"),
    ("untap_land", r"\buntap\b.*\bland\b"),
    
    # tutors / cheat
    ("search_library", r"\bsearch your library\b"),
    ("onto_battlefield", r"\bonto the battlefield\b"),
    ("from_graveyard", r"\bfrom (your )?graveyard\b"),
    ("graveyard_to_battlefield", r"\bfrom (your )?graveyard\b.*\bbattlefield\b"),
    
    # board-wide / scaling
    ("each", r"\beach\b"),
    ("all", r"\ball\b"),
    ("choose_one", r"\bchoose one\b"),
    ("choose_two", r"\bchoose two\b"),
    ("up_to", r"\bup to\b"),
    ("for_each", r"\bfor each\b"),
    ("equal_to", r"\bequal to\b"),
    ("has_x", r"\bwhere x is\b|\bx\b"),
    
    # tokens
    ("create_token", r"\bcreate\b.*\btoken\b"),
    
    # keywords (important for CMC)
    ("flying", r"\bflying\b"),
    ("trample", r"\btrample\b"),
    ("haste", r"\bhaste\b"),
    ("vigilance", r"\bvigilance\b"),
    ("lifelink", r"\blifelink\b"),
    ("deathtouch", r"\bdeathtouch\b"),
    ("first_strike", r"\bfirst strike\b"),
    ("double_strike", r"\bdouble strike\b"),
    ("reach", r"\breach\b"),
    ("menace", r"\bmenace\b"),
    ("flash", r"\bflash\b"),
    ("hexproof", r"\bhexproof\b"),
    ("indestructible", r"\bindestructible\b"),
    ("ward", r"\bward\b"),
    ("defender", r"\bdefender\b"),
    ("prowess", r"\bprowess\b"),
    
    # subtype/equipment cues
    ("equip", r"\bequip\b"),
    ("enchant", r"\benchant\b"),
    ("crew", r"\bcrew\b"),
    ("attach", r"\battach\b"),
    
    # triggers
    ("enters_battlefield", r"\benters\b.*\bbattlefield\b"),
    ("dies", r"\bdies\b"),
    ("attack", r"\battacks?\b"),
    ("block", r"\bblocks?\b"),
    ("beginning_upkeep", r"\bbeginning of\b.*\bupkeep\b"),
    ("end_step", r"\bend step\b"),
    
    # protection / evasion
    ("cant_be_blocked", r"\bcan't be blocked\b"),
    ("protection", r"\bprotection\b"),
    ("shroud", r"\bshroud\b"),
    
    # counters
    ("plus_counter", r"\+1/\+1 counter"),
    ("minus_counter", r"\-1/\-1 counter"),
    ("loyalty_counter", r"\bloyalty counter\b"),
]


def stable_hash_to_index(token: str, dim: int) -> int:
    h = hashlib.md5(token.encode("utf-8", errors="ignore")).hexdigest()
    return int(h, 16) % dim


def safe_literal_list(x) -> List[str]:
    """Parse values like "['W','U']" into list[str]."""
    if x is None:
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
        if s:
            return [s]
    return []


def to_float_or_nan(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return float("nan")
        try:
            return float(s)
        except Exception:
            return float("nan")
    return float("nan")


def normalize_type_line(type_line: Optional[str]) -> str:
    if not type_line:
        return ""
    # Handle common mojibake variants of an em dash.
    return (
        str(type_line)
        .replace("â€”", "—")
        .replace("â€“", "—")
        .replace("—", " — ")
        .strip()
    )


def extract_type_tokens(type_line: str) -> Tuple[List[str], List[str]]:
    """Return (types, subtypes) lowercased tokens from a type line."""
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


def regex_count(pat: re.Pattern, text: str) -> int:
    return len(pat.findall(text))


@dataclass
class FeatureConfig:
    hash_dim: int = 256
    use_text_indicators: bool = True
    include_set_token: bool = False  # can overfit to printings/era


@dataclass
class Normalizer:
    mean: List[float]
    std: List[float]


def build_features_from_row(
    row_vals: Tuple,
    idx: dict,
    cfg: FeatureConfig,
    compiled_patterns: List[Tuple[str, re.Pattern]],
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      - hashed categorical vector (hash_dim,)
      - numeric engineered vector (K,)
      - cmc (int)
    """
    def get(col, default=None):
        j = idx.get(col)
        if j is None:
            return default
        v = row_vals[j]
        return v if v is not None else default

    # Target
    cmc_raw = get("cmc", None)
    if cmc_raw is None:
        raise ValueError("missing cmc")
    cmc = int(round(float(cmc_raw)))
    if cmc < 0:
        raise ValueError("negative cmc")

    # ---- hashed categorical tokens
    hashed = np.zeros((cfg.hash_dim,), dtype=np.float32)

    type_line = str(get("type", "") or "")
    types, subtypes_from_type = extract_type_tokens(type_line)

    for t in types:
        hashed[stable_hash_to_index(f"type:{t}", cfg.hash_dim)] += 1.0

    # union explicit subtypes column with parsed type-line subtypes
    subtype_list = [s.lower() for s in safe_literal_list(get("subtypes", None))]
    all_subtypes = list(dict.fromkeys(subtypes_from_type + subtype_list))
    for st in all_subtypes:
        if st:
            hashed[stable_hash_to_index(f"subtype:{st}", cfg.hash_dim)] += 1.0

    for st in [s.lower() for s in safe_literal_list(get("supertypes", None))]:
        if st:
            hashed[stable_hash_to_index(f"super:{st}", cfg.hash_dim)] += 1.0

    layout = str(get("layout", "") or "").strip().lower()
    if layout:
        hashed[stable_hash_to_index(f"layout:{layout}", cfg.hash_dim)] += 1.0

    rarity = str(get("rarity", "") or "").strip().lower()
    if rarity:
        hashed[stable_hash_to_index(f"rarity:{rarity}", cfg.hash_dim)] += 1.0

    # colors: prefer color_identity if present
    colors = safe_literal_list(get("color_identity", None))
    if not colors:
        colors = safe_literal_list(get("colors", None))
    colors = [c.strip().upper() for c in colors if c]
    for c in colors:
        hashed[stable_hash_to_index(f"color:{c}", cfg.hash_dim)] += 1.0
    if len(colors) == 0:
        hashed[stable_hash_to_index("color:COLORLESS", cfg.hash_dim)] += 1.0

    if cfg.include_set_token:
        set_code = str(get("set", "") or "").strip().lower()
        if set_code:
            hashed[stable_hash_to_index(f"set:{set_code}", cfg.hash_dim)] += 1.0

    # ---- numeric engineered features (missingness-aware)
    power = to_float_or_nan(get("power", None))
    toughness = to_float_or_nan(get("toughness", None))
    loyalty = to_float_or_nan(get("loyalty", None))

    power_missing = float(math.isnan(power))
    toughness_missing = float(math.isnan(toughness))
    loyalty_missing = float(math.isnan(loyalty))

    power = 0.0 if math.isnan(power) else power
    toughness = 0.0 if math.isnan(toughness) else toughness
    loyalty = 0.0 if math.isnan(loyalty) else loyalty

    is_creature = 1.0 if "creature" in types else 0.0
    is_planeswalker = 1.0 if "planeswalker" in types else 0.0
    num_colors = float(len(colors))

    text = str(get("text", "") or "")
    text_lc = text.lower()

    # simple complexity proxies
    word_count = float(len(re.findall(r"\w+", text_lc)))
    line_count = float(text_lc.count("\n") + (1 if text_lc else 0))
    digit_count = float(len(re.findall(r"\d", text_lc)))
    mana_symbol_in_text = float(text_lc.count("{"))

    # planeswalker ability line count (+ / - / 0 line starts)
    pw_ability_lines = 0.0
    if is_planeswalker and text_lc:
        pw_ability_lines = float(len(re.findall(r"(?m)^[+\-0]", text_lc)))

    indicator_vals: List[float] = []
    if cfg.use_text_indicators:
        for _name, pat in compiled_patterns:
            indicator_vals.append(float(regex_count(pat, text_lc)))
    else:
        indicator_vals = [0.0] * len(compiled_patterns)

    numeric = np.array(
        [
            is_creature,
            is_planeswalker,
            num_colors,
            power,
            toughness,
            loyalty,
            power_missing,
            toughness_missing,
            loyalty_missing,
            word_count,
            line_count,
            digit_count,
            mana_symbol_in_text,
            pw_ability_lines,
            *indicator_vals,
        ],
        dtype=np.float32,
    )

    return hashed, numeric, cmc


def expected_value_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    classes = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
    return (probs * classes[None, :]).sum(dim=1)


def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = (counts.sum() / (counts + 1e-9))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def fit_normalizer(x_num_train: np.ndarray) -> Normalizer:
    mean = x_num_train.mean(axis=0)
    std = x_num_train.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return Normalizer(mean=mean.tolist(), std=std.tolist())


def apply_normalizer(x_num: np.ndarray, norm: Normalizer) -> np.ndarray:
    mean = np.array(norm.mean, dtype=np.float32)
    std = np.array(norm.std, dtype=np.float32)
    return (x_num - mean) / std


class MLPClassifier(nn.Module):
    """
    Larger architecture for CMC prediction with LR scheduling.
    768 → 512 → 256 → 128 → n_classes
    """
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def open_sheet(path: str, sheet_name: Optional[str] = None):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name] if sheet_name else wb[wb.sheetnames[0]]
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    headers = [h for h in header if h is not None]
    idx = {h: i for i, h in enumerate(headers)}
    return wb, ws, headers, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=str, default="all_mtg_cards (2).xlsx", help="Path to the MTG cards Excel file")
    ap.add_argument("--out-prefix", type=str, default="mtg_cmc_dl")
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--hash-dim", type=int, default=1024)
    ap.add_argument("--no-text-indicators", action="store_true")
    ap.add_argument("--max-cmc", type=int, default=10, help="Maximum CMC to include (filters outliers)")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.max_cmc != 10:
        raise ValueError("Universal benchmark requires --max-cmc 10")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = FeatureConfig(
        hash_dim=args.hash_dim,
        use_text_indicators=not args.no_text_indicators,
        include_set_token=False,
    )
    compiled_patterns = [(name, re.compile(pat)) for name, pat in EFFECT_PATTERNS]

    # The saved manifest defines eligible rows, labels, and all three partitions.
    split = load_shared_split(Path(args.xlsx), args.split_file, args.seed)
    y = split.target
    valid_count = len(y)
    max_cmc = 10
    n_classes = 11
    positions = {int(raw_row): position for position, raw_row in enumerate(split.source_rows)}
    x_hash = np.zeros((valid_count, cfg.hash_dim), dtype=np.float32)
    x_num = None
    seen = np.zeros(valid_count, dtype=bool)
    wb, ws, headers, idx = open_sheet(args.xlsx)
    try:
        for raw_row, row in enumerate(ws.iter_rows(min_row=2, values_only=True)):
            position = positions.get(raw_row)
            if position is None:
                continue
            h, numeric, cmc = build_features_from_row(
                tuple(row[:len(headers)]), idx, cfg, compiled_patterns)
            if cmc != y[position]:
                raise ValueError(f"CMC mismatch at spreadsheet row {raw_row + 2}")
            if x_num is None:
                x_num = np.zeros((valid_count, len(numeric)), dtype=np.float32)
            x_hash[position] = h
            x_num[position] = numeric
            seen[position] = True
    finally:
        wb.close()
    if not seen.all() or x_num is None:
        raise ValueError("Feature extraction did not cover every row in the shared split")
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx
    y_train = y[train_idx]

    # -------- normalize numeric features (train-only fit)
    norm = fit_normalizer(x_num[train_idx])
    x_num_norm = apply_normalizer(x_num, norm)

    x_all = np.concatenate([x_hash, x_num_norm], axis=1).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Samples: {valid_count} | Train: {len(train_idx)} | "
          f"Val: {len(val_idx)} | Test: {len(test_idx)}")

    def make_loader(indices, shuffle):
        xt = torch.tensor(x_all[indices], dtype=torch.float32)
        yt = torch.tensor(y[indices], dtype=torch.long)
        ds = torch.utils.data.TensorDataset(xt, yt)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)
    test_loader = make_loader(test_idx, shuffle=False)

    model = MLPClassifier(in_dim=x_all.shape[1], n_classes=n_classes).to(device)
    class_w = compute_class_weights(y_train, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler - reduce LR when validation plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    best_val_mae = float("inf")
    best_state = None
    patience = 200
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
        train_loss /= len(train_idx)

        model.eval()
        val_loss = 0.0
        abs_err = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                pred_ev = expected_value_from_logits(logits)
                abs_err += torch.abs(pred_ev - yb.float()).sum().item()
                n += xb.size(0)

        val_loss /= len(val_idx)
        val_mae = abs_err / max(n, 1)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_MAE(expected)={val_mae:.3f}")
        
        # Step the scheduler
        scheduler.step(val_mae)

        if val_mae + 1e-6 < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # The test set is evaluated only after checkpoint selection on validation.
    model.eval()
    print(f"Training complete. Best validation expected-value MAE: {best_val_mae:.3f}")

    test_probabilities = []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            test_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    test_metrics, test_predicted = score_probabilities(
        np.concatenate(test_probabilities), y[test_idx], split.groups[test_idx])
    test_metrics.update({"feature_count": int(x_all.shape[1]),
                         "checkpoint_selection": "lowest validation expected-value MAE"})
    print(f"Test | within 1={test_metrics['within_one_accuracy']:.3%} | "
          f"class MAE={test_metrics['class_mae']:.3f}")
    save_benchmark_result(split, args.output_dir, "expanded_mlp", test_metrics, test_predicted)

    # -------- save
    model_path = f"{args.out_prefix}_model.pt"
    meta_path = f"{args.out_prefix}_meta.json"
    torch.save(model.state_dict(), model_path)

    meta = {
        "hash_dim": cfg.hash_dim,
        "use_text_indicators": cfg.use_text_indicators,
        "effect_patterns": [name for name, _ in EFFECT_PATTERNS],
        "normalizer": asdict(norm),
        "n_classes": n_classes,
        "max_cmc": max_cmc,
        "numeric_feature_order": [
            "is_creature",
            "is_planeswalker",
            "num_colors",
            "power",
            "toughness",
            "loyalty",
            "power_missing",
            "toughness_missing",
            "loyalty_missing",
            "word_count",
            "line_count",
            "digit_count",
            "mana_symbol_in_text",
            "pw_ability_lines",
            *[f"cnt_{name}" for name, _ in EFFECT_PATTERNS],
        ],
        "metrics": {"best_val_mae": best_val_mae, "test": test_metrics},
        "split_sha256": split.split_sha256,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", model_path)
    print("Saved:", meta_path)


if __name__ == "__main__":
    main()
