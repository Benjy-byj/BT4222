import argparse
import sys
import re
import io
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd

# ----------------- Sheet name helpers -----------------
INVALID_SHEET_CHARS = r'[:\\/?*\[\]]'
MAX_SHEET_LEN = 31

def sanitize_sheet_name(name: str) -> str:
    base = re.sub(INVALID_SHEET_CHARS, "_", name)
    base = base.strip()
    if not base:
        base = "Sheet"
    if len(base) > MAX_SHEET_LEN:
        base = base[:MAX_SHEET_LEN]
    return base

def make_unique(name: str, used: set) -> str:
    candidate = name
    n = 2
    while candidate in used:
        suffix = f"({n})"
        max_base_len = MAX_SHEET_LEN - len(suffix)
        candidate = (name[:max_base_len]).rstrip() + suffix
        n += 1
    used.add(candidate)
    return candidate

# ----------------- CSV sniffing & reading -----------------
DELIMS_TO_TRY = [",", ";", "\t", "|"]
ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "gb18030", "latin1"]

def _read_head_bytes(p: Path, n_bytes: int = 65536) -> bytes:
    with open(p, "rb") as f:
        return f.read(n_bytes)

def _try_decode(b: bytes, encoding: str) -> Optional[str]:
    try:
        return b.decode(encoding)
    except Exception:
        return None

def _best_delim_and_header(lines: List[str]) -> Tuple[str, int]:
    best = (",", 0, 1)
    for d in DELIMS_TO_TRY:
        col_counts = [len(l.rstrip("\n\r").split(d)) for l in lines]
        max_cols = max(col_counts) if col_counts else 1
        if max_cols <= 1:
            continue
        header_idx = next((i for i, c in enumerate(col_counts) if c == max_cols), 0)
        if (max_cols > best[2]) or (max_cols == best[2] and header_idx < best[1]):
            best = (d, header_idx, max_cols)
    return best[0], best[1]

def sniff_csv(p: Path) -> Tuple[str, str, int]:
    raw = _read_head_bytes(p)
    chosen_enc = None
    text = None
    for enc in ENCODINGS_TO_TRY:
        text = _try_decode(raw, enc)
        if text is not None:
            chosen_enc = enc
            break
    if chosen_enc is None:
        chosen_enc = "latin1"
        text = raw.decode("latin1", errors="replace")

    lines = []
    sample_io = io.StringIO(text)
    for _ in range(120):
        line = sample_io.readline()
        if not line:
            break
        lines.append(line)

    delim, header_idx = _best_delim_and_header(lines)
    skiprows = header_idx
    return chosen_enc, delim, skiprows

def read_csv_robust(p: Path) -> pd.DataFrame:
    enc, sep, skiprows = sniff_csv(p)
    try:
        return pd.read_csv(p, encoding=enc, sep=sep, engine="python",
                           skiprows=skiprows, header=0)
    except Exception:
        # fallback brute force
        for enc2 in ENCODINGS_TO_TRY:
            for sep2 in DELIMS_TO_TRY:
                try:
                    return pd.read_csv(p, encoding=enc2, sep=sep2, engine="python")
                except Exception:
                    continue
        raise

# ----------------- Excel reading -----------------
def read_excel_sheets(p: Path, include_all: bool) -> List[Tuple[str, pd.DataFrame]]:
    try:
        xls = pd.ExcelFile(p)
    except ImportError as e:
        raise e
    except Exception as e:
        raise e

    out = []
    if include_all:
        for sh in xls.sheet_names:
            try:
                df = xls.parse(sh)
                out.append((sh, df))
            except Exception as e:
                print(f"[WARN] Failed to read sheet '{sh}' in '{p.name}': {e}")
    else:
        try:
            df = xls.parse(0)
            out.append((xls.sheet_names[0] if xls.sheet_names else 'Sheet1', df))
        except Exception as e:
            print(f"[WARN] Failed to read first sheet in '{p.name}': {e}")
    return out

# ----------------- Main merging logic -----------------
def merge_dir_to_workbook(input_dir: Path, output_file: Path, include_all_xlsx_sheets: bool):
    files = sorted(
        [p for p in input_dir.glob("**/*") if p.suffix.lower() in [".csv", ".xlsx", ".xls"]],
        key=lambda x: x.name.lower()
    )
    if not files:
        print(f"No CSV/XLSX/XLS files found under: {input_dir}")
        sys.exit(1)

    used_names = set()

    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        for p in files:
            # ---- Fix: ensure try has a matching except ----
            try:
                if p.suffix.lower() == ".csv":
                    df = read_csv_robust(p)
                    base = sanitize_sheet_name(p.stem)
                    sheet_name = make_unique(base, used_names)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                else:  # xlsx/xls
                    try:
                        sheets = read_excel_sheets(p, include_all=include_all_xlsx_sheets)
                    except ImportError as ie:
                        msg = str(ie)
                        if "openpyxl" in msg.lower():
                            print(f"[WARN] '{p}': need openpyxl (pip install openpyxl)")
                        elif "xlrd" in msg.lower():
                            print(f"[WARN] '{p}': need xlrd for .xls (pip install xlrd)")
                        else:
                            print(f"[WARN] '{p}': {ie}")
                        continue
                    except Exception as e:
                        print(f"[WARN] Failed to open '{p}': {e}")
                        continue

                    for sh_name, df in sheets:
                        base = sanitize_sheet_name(
                            f"{p.stem}_{sh_name}" if include_all_xlsx_sheets else p.stem
                        )
                        sheet_name = make_unique(base, used_names)
                        try:
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                        except Exception as e:
                            print(f"[WARN] Failed to write '{sheet_name}' from '{p.name}': {e}")
            except Exception as e:
                # catch any unexpected error in this file and continue with next file
                print(f"[WARN] Failed to add '{p}': {e}")
                continue
            # ---- End fix ----

    print(f"✅ Merged {len(files)} file(s) into: {output_file}")

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple CSV/XLSX/XLS into one Excel workbook, one sheet per file (or per original sheet)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input directory")
    parser.add_argument("--output", required=True, type=Path, help="Output Excel file, e.g., merged.xlsx")
    parser.add_argument("--all-xlsx-sheets", action="store_true",
                        help="Include ALL sheets from each Excel file; default: only the first sheet.")
    args = parser.parse_args()

    merge_dir_to_workbook(args.input, args.output, include_all_xlsx_sheets=args.all_xlsx_sheets)

if __name__ == "__main__":
    main()
