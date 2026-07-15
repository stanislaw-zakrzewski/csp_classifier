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
