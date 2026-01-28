#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
import numpy as np

def load_meta(meta_path: Path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict) or "stoi" not in meta:
        raise RuntimeError("meta.pkl must be a dict containing 'stoi'")
    return meta

def load_bin_memmap(bin_path: Path):
    return np.memmap(bin_path, dtype=np.uint16, mode="r")

def find_starts_by_token(data: np.ndarray, token_id: int):
    # positions where data[i] == token_id
    starts = np.where(data == np.uint16(token_id))[0].astype(np.int64)
    if starts.size:
        starts = np.unique(starts)
        starts.sort()
    return starts

def build_ends(starts: np.ndarray, total_len: int):
    ends = np.empty_like(starts)
    if starts.size == 0:
        return ends
    ends[:-1] = starts[1:]
    ends[-1] = total_len
    return ends

def slice_by_first_n(data: np.ndarray, starts: np.ndarray, n: int, min_len: int):
    if starts.size < n:
        raise RuntimeError(f"Not enough samples: found {starts.size}, need {n}")
    ends = build_ends(starts, len(data))
    sel_s = starts[:n]
    sel_e = ends[:n]

    # filter by min_len while preserving order
    kept_s = []
    kept_e = []
    for s, e in zip(sel_s, sel_e):
        if e - s >= min_len:
            kept_s.append(int(s))
            kept_e.append(int(e))
    if len(kept_s) < n:
        # If many too-short segments, extend window until we have n
        for s, e in zip(starts[n:], ends[n:]):
            if len(kept_s) >= n:
                break
            if e - s >= min_len:
                kept_s.append(int(s)); kept_e.append(int(e))
    if len(kept_s) < n:
        raise RuntimeError(f"After min_len filter, only kept {len(kept_s)} < {n}. Try lower --min_len.")

    kept_s = np.array(kept_s[:n], dtype=np.int64)
    kept_e = np.array(kept_e[:n], dtype=np.int64)

    total = int(np.sum(kept_e - kept_s))
    out = np.empty((total,), dtype=data.dtype)
    cur = 0
    for s, e in zip(kept_s, kept_e):
        seg = np.asarray(data[s:e])
        out[cur:cur+len(seg)] = seg
        cur += len(seg)
    return out, kept_s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="dir containing train.bin/val.bin/meta.pkl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=9000)
    ap.add_argument("--n_val", type=int, default=1000)
    ap.add_argument("--min_len", type=int, default=32, help="min token length per sample segment")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(in_dir / "meta.pkl")
    stoi = meta["stoi"]

    if "data_" not in stoi:
        # help debug tokenization variants
        cand = [k for k in stoi.keys() if "data" in str(k).lower()]
        raise RuntimeError(f"'data_' not found in stoi. data-like tokens: {cand[:30]}")

    data_tok = int(stoi["data_"])
    print("[info] token id for 'data_':", data_tok)

    train = load_bin_memmap(in_dir / "train.bin")
    val   = load_bin_memmap(in_dir / "val.bin")

    train_starts = find_starts_by_token(train, data_tok)
    val_starts   = find_starts_by_token(val, data_tok)

    print(f"[info] found train starts: {len(train_starts)}  val starts: {len(val_starts)}")

    train_out, train_kept_starts = slice_by_first_n(train, train_starts, args.n_train, args.min_len)
    val_out, val_kept_starts     = slice_by_first_n(val, val_starts, args.n_val, args.min_len)

    (out_dir / "train.bin").write_bytes(train_out.tobytes())
    (out_dir / "val.bin").write_bytes(val_out.tobytes())

    meta_out = dict(meta)
    meta_out["subset_note"] = {
        "method": "scan token stream for token 'data_' and slice by segments [data_i, data_{i+1})",
        "n_train": args.n_train,
        "n_val": args.n_val,
        "min_len": args.min_len,
        "source_dir": str(in_dir),
        "data_token": "data_",
        "data_token_id": data_tok,
        "found_train_starts": int(len(train_starts)),
        "found_val_starts": int(len(val_starts)),
    }
    meta_out["subset_train_starts"] = train_kept_starts
    meta_out["subset_val_starts"] = val_kept_starts

    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta_out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[OK] wrote:", out_dir)
    print(" train.bin bytes:", (out_dir / "train.bin").stat().st_size)
    print(" val.bin bytes:", (out_dir / "val.bin").stat().st_size)
    print(" meta.pkl bytes:", (out_dir / "meta.pkl").stat().st_size)

if __name__ == "__main__":
    main()
