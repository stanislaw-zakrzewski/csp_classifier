# Windows MOABB Path Sanitization Issue

This document explains the issue where MOABB on Windows downloads files to a directory named `D-` (or `C-`) instead of the correct drive (e.g. `D:\` or `C:\`), and how to patch it.

## The Problem
In the `moabb` library (specifically in `moabb/datasets/download.py`), the function `_sanitize_path` is designed to strip invalid characters from paths by translating them to hyphens. 

The original code is:
```python
def _sanitize_path(path: Path) -> Path:
    table = {ord(c): "-" for c in ':*?"<>|'}
    return Path(str(path).translate(table))
```

On Windows, absolute paths contain a drive letter followed by a colon (e.g., `D:`). Since the colon (`:`) is in the sanitization character table, `_sanitize_path` converts the drive letter prefix `D:` to `D-`. This breaks the drive anchor, forcing Python to resolve the rest of the path as a relative path under the current working directory, for example:
`C:\Users\stz\Documents\GitHub\csp_classifier\D-\EEG_DATA\...`

## The Solution
To fix this, we modify `_sanitize_path` to keep the drive anchor (if it exists) untouched, and only sanitize the relative portion of the path.

### How to Patch
Locate `moabb/datasets/download.py` in your virtual environment:
`venv/Lib/site-packages/moabb/datasets/download.py`

Modify the `_sanitize_path` function to look as follows:

```python
def _sanitize_path(path: Path) -> Path:
    table = {ord(c): "-" for c in ':*?"<>|'}
    anchor = path.anchor
    if anchor:
        try:
            rel = path.relative_to(anchor)
            return Path(anchor) / str(rel).translate(table)
        except ValueError:
            pass
    return Path(str(path).translate(table))
```

This ensures that drive anchors (like `C:\` or `D:\` on Windows, or `/` on Linux) are preserved intact, while only subsequent directory names and files undergo sanitization.

---

# Yang2025 Dataset Infinite Zip Extraction Issue

## The Problem
In `moabb/datasets/yang2025.py`, the `data_path(subject)` method checks whether data for a subject already exists before triggering download or extraction:
```python
subj_str = f"sub-{subject:03d}"
existing = list(basepath.rglob(f"*{subj_str}*data.bdf"))
```
Because the downloaded archive unpacks BDF files named simply `data.bdf` located within subdirectories named `sub-001`, `sub-002`, etc., matching the filename against `*{subj_str}*data.bdf` using `Path.rglob` always returns an empty list (`[]`).

As a result, even after downloading and extracting the dataset, MOABB attempts to re-extract the entire 65.6 GB `WBCIC_SHU_Motor_Imagery_dataset.zip` archive on **every single call** to `get_data()`. In Python, extracting a 65.6 GB zip archive single-threaded takes tens of minutes to hours, making the execution appear completely stuck at `Loading dataset Yang2025 for subject...`.

## The Solution
Locate `moabb/datasets/yang2025.py` in your virtual environment:
`venv/Lib/site-packages/moabb/datasets/yang2025.py`

Modify lines 346–350 to check for `subj_str` in directory path parts rather than matching only the filename:

```python
# Check if data already exists (raw BDF files)
subj_str = f"sub-{subject:03d}"
existing = [
    p for p in basepath.rglob("data.bdf")
    if any(subj_str in part for part in p.parts)
]
if existing:
    return str(basepath)
```

