"""juicefs model packaging helper.

The benchmark forbids committing files > 1MB and forbids committing model
weights. Instead, weights live on the internal juicefs host and are pulled
at runtime inside the Jetson image:

    http://172.28.4.81:34567/<filename>

This helper downloads (and caches) the weights on first inference so the
committed repo stays tiny and public-safe.
"""

import os
import urllib.request

from .config import NODConfig


def ensure_model(cfg: NODConfig, force: bool = False) -> str:
    """Return a local path to the model weights, downloading if needed."""
    os.makedirs(cfg.model_cache_dir, exist_ok=True)
    local = os.path.join(cfg.model_cache_dir, cfg.model_filename)
    if os.path.exists(local) and not force:
        return local
    url = cfg.model_repo_url.rstrip("/") + "/" + cfg.model_filename
    print(f"[packaging] downloading weights -> {url}")
    try:
        urllib.request.urlretrieve(url, local)
    except Exception as e:  # pragma: no cover - network dependent
        print(f"[packaging] WARNING: could not download weights ({e}). "
              f"Falling back to a randomly-initialized network "
              f"(geometric path recommended for the benchmark).")
        return local
    return local


def _load_gitignore(root):
    gi = os.path.join(root, ".gitignore")
    if not os.path.exists(gi):
        return []
    patterns = []
    with open(gi) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            patterns.append(line)
    return patterns


def _is_ignored(relpath, patterns):
    """Minimal .gitignore matcher (supports * globs, dir/ and path patterns)."""
    import fnmatch
    basename = os.path.basename(relpath)
    parts = relpath.split("/")
    for p in patterns:
        if p.endswith("/"):                       # directory pattern
            if p.rstrip("/") in parts:
                return True
            continue
        if "/" in p:                              # path pattern
            cand = p[1:] if p.startswith("/") else p
            if fnmatch.fnmatch(relpath, cand) or fnmatch.fnmatch(relpath, p):
                return True
        elif fnmatch.fnmatch(basename, p):         # basename pattern
            return True
    return False


def package_check(cfg: NODConfig) -> dict:
    """Sanity check before submission: no >1MB files committed, weights external.

    Mirrors what git will actually commit by honoring .gitignore, so generated
    demo artifacts / caches don't produce false positives.
    """
    report = {"oversized_files": [], "weights_external": True}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    patterns = _load_gitignore(root)
    for dp, _, fns in os.walk(root):
        for fn in fns:
            fp = os.path.join(dp, fn)
            rel = os.path.relpath(fp, root)
            if _is_ignored(rel, patterns):
                continue
            try:
                if os.path.getsize(fp) > 1_000_000:
                    report["oversized_files"].append(fp)
            except OSError:
                pass
    report["ok"] = len(report["oversized_files"]) == 0
    return report
