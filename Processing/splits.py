from __future__ import annotations
import re
import os
from pathlib import Path
import shutil
import json
import csv

# ---- config ----
SRC_DIR = Path("../Data/Raw")
OUT_DIR = Path("../Data/Splits")
COPY_MODE = "copy"   # "copy" or "symlink"

OUT_DIRS = {
    "uninjured_control": OUT_DIR / "uninjured_control",
    "uninjured_aclr":    OUT_DIR / "uninjured_aclr",
    "injured_aclr":      OUT_DIR / "injured_aclr",
}

# Example names:
#   1_Left_ACLR_1_mvic1.csv  -> ACLR, flag 1 (injured)
#   1_Left_ACLR_0_mvic1.csv  -> ACLR, flag 0 (uninjured)
#   5_Left_CONT_1_mvic1.csv  -> CONT (control; uninjured)
# We’ll accept flexible variants and be robust to extra tokens before the extension.

FNAME_RE = re.compile(
    r"""
    ^
    (?P<session>\d+)_
    (?P<side>Left|Right)_
    (?P<group>ACLR|CONT)_
    (?P<flag>\d+)
    _mvic(?P<trial>\d+)
    (?:\.[A-Za-z0-9]+)?      # extension
    $
    """,
    re.VERBOSE | re.IGNORECASE
)

def parse_filename(name: str) -> Optional[dict]:
    m = FNAME_RE.match(name)
    if not m:
        # Try without extension if it failed due to double extensions etc.
        m = FNAME_RE.match(Path(name).stem)
        if not m:
            return None
    d = m.groupdict()
    session = int(d["session"])
    side    = d["side"].capitalize()
    group   = d["group"].upper()
    flag    = int(d["flag"])
    trial   = int(d["trial"])

    # Decide label bucket
    if group == "CONT":
        label = "uninjured_control"
        injured = 0
    elif group == "ACLR":
        if flag == 1:
            label = "injured_aclr"
            injured = 1
        elif flag == 0:
            label = "uninjured_aclr"
            injured = 0
        else:
            raise ValueError(f"ACLR flag must be 0/1 in: {name}")
    else:
        raise ValueError(f"Unknown group: {group} in {name}")

    return {
        "session": session,
        "side": side,
        "group": group,
        "flag": flag,
        "trial": trial,
        "injured": injured,
        "label": label,
    }

def ensure_dirs():
    for p in OUT_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

def place_file(src: Path, label: str) -> Path:
    dst_dir = OUT_DIRS[label]
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if COPY_MODE == "copy":
        shutil.copy2(src, dst)
    elif COPY_MODE == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(os.path.relpath(src, dst_dir), dst)
    else:
        raise ValueError("COPY_MODE must be 'copy' or 'symlink'")
    return dst

def build_manifests(records: list[dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(records, f, indent=2)
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","label","session","trial","side","group","flag","injured","src_path"])
        for r in records:
            w.writerow([r["dst_path"], r["label"], r["session"], r["trial"], r["side"], r["group"], r["flag"], r["injured"], r["src_path"]])

def split_all():
    ensure_dirs()
    records: list[dict] = []

    # recursive scan
    for p in sorted(SRC_DIR.rglob("*.csv")):
        meta = parse_filename(p.name)
        if meta is None:
            print(f"[skip] pattern mismatch: {p}")
            continue
        try:
            dst = place_file(p, meta["label"])
        except Exception as e:
            print(f"[error] {p.name}: {e}")
            continue
        rec = {"src_path": str(p), "dst_path": str(dst), **meta}
        records.append(rec)

    build_manifests(records)

    # summary
    print("Done.")
    for label, d in OUT_DIRS.items():
        count = len(list(d.glob("*.csv")))
        print(f"  {label:18s}: {count:4d} files -> {d}")
    # optional: session/trial coverage
    sessions = sorted({r["session"] for r in records})
    trials   = sorted({r["trial"] for r in records})
    print(f"  sessions found: {sessions}")
    print(f"  trials found:   {trials}")

if __name__ == "__main__":
    split_all()