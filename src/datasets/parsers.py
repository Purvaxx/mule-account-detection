import os
import hashlib
from pathlib import Path
import pandas as pd
from typing import Tuple

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

def safe_ingest(workspace_dir: Path, source_path: str) -> Tuple[Path, str, int]:
    """
    Safely copy an uploaded file into the workspace directory, hashing it and
    verifying its extension.
    """
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    ext = source.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported extension {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    if ext in [".xlsx", ".xls"]:
        import pandas as pd
        with pd.ExcelFile(source) as xl:
            if len(xl.sheet_names) > 1:
                raise ValueError(f"Multi-sheet Excel files are not supported. Please upload a single-sheet workbook.")

    # Generate a safe local filename in the workspace
    dest = workspace_dir / f"raw_data{ext}"
    
    # Hash and copy in chunks for large file safety
    sha256 = hashlib.sha256()
    size = 0
    with open(source, "rb") as f_in, open(dest, "wb") as f_out:
        while chunk := f_in.read(8192):
            size += len(chunk)
            sha256.update(chunk)
            f_out.write(chunk)
            
    if size == 0:
        # Cleanup and raise
        dest.unlink(missing_ok=True)
        raise ValueError("Uploaded file is empty.")
        
    return dest, sha256.hexdigest(), size

def load_sample(file_path: Path, max_rows: int = 1000) -> pd.DataFrame:
    """
    Safely load a bounded sample for profiling without executing macros or
    loading giant files entirely into memory.
    """
    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path, nrows=max_rows, on_bad_lines='skip')
        elif ext in [".xlsx", ".xls"]:
            # openpyxl for xlsx, xlrd for xls (pandas handles this)
            # We enforce nrows to limit memory consumption
            return pd.read_excel(file_path, nrows=max_rows)
        else:
            raise ValueError(f"Unsupported extension {ext}")
    except Exception as e:
        raise ValueError(f"Failed to parse {ext} file: {e}")


def load_full(file_path: Path) -> pd.DataFrame:
    """Load a workspace file in its entirety, for canonicalization.

    The counterpart to :func:`load_sample`: this module owns *how a workspace
    file becomes a DataFrame*, so canonicalization does not re-implement the
    extension dispatch and drift from it.

    Unlike the profiling sample this is deliberately **strict** — a malformed
    CSV row raises rather than being skipped.  A bounded sample exists to
    describe a file's shape and can tolerate losing a row; the canonical frame
    becomes the analysed dataset, and silently dropping records from it would
    change what the model is scored on without saying so.
    """
    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path, low_memory=False)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported extension {ext}")
    except Exception as e:
        raise ValueError(f"Failed to parse {ext} file: {e}")
