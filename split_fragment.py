def make_splits(a, paths, y, classes):
    """Keep identical decoded RGB images and optional plant groups in one fold.

    groups_csv must contain path,group columns and cover every training image.
    Paths are relative to a.train. This detects exact decoded-pixel duplicates,
    not near duplicates or different photographs of the same plant; use group
    metadata for those cases. Image decoding matches PathDataset (no EXIF rotate).
    """
    import csv
    import hashlib
    from pathlib import Path, PureWindowsPath

    import numpy as np
    from PIL import Image
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    paths = [Path(p).resolve() for p in paths]
    labels = np.asarray(y, dtype=np.int64)
    n = len(paths)
    expected_labels = set(range(len(classes)))
    if n != len(labels) or not n:
        raise ValueError("Training paths and labels must be nonempty and have equal length.")
    if set(labels.tolist()) != expected_labels:
        raise ValueError("Training labels must cover every class in classes, using contiguous IDs.")
    if a.folds < 2:
        raise ValueError("Need at least two folds.")
    if len(set(paths)) != n:
        raise ValueError("Training image list contains the same path more than once.")
    train_root = Path(a.train).resolve()
    try:
        relative_paths = [p.relative_to(train_root).as_posix() for p in paths]
    except ValueError as exc:
        raise ValueError("Every training image must be inside the train directory.") from exc
    index_by_path = {p: i for i, p in enumerate(paths)}
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    digest_first = {}
    digests = []
    print(f"Checking exact RGB duplicates in {n} training images...", flush=True)
    for i, path in enumerate(paths):
        try:
            with Image.open(path) as im:
                rgb = im.convert("RGB")
                h = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
                h.update(rgb.tobytes())
                digest = h.hexdigest()
        except Exception as exc:
            raise ValueError(f"Cannot decode training image: {path}") from exc
        digests.append(digest)
        if digest in digest_first:
            j = digest_first[digest]
            if labels[i] != labels[j]:
                raise ValueError(
                    "Identical RGB images have conflicting labels: "
                    f"{relative_paths[j]} ({classes[labels[j]]}) and "
                    f"{relative_paths[i]} ({classes[labels[i]]}). Correct these labels first.")
            union(i, j)
        else:
            digest_first[digest] = i
        if (i + 1) % 500 == 0:
            print(f"  hashed {i + 1}/{n}", flush=True)

    supplied_groups = [""] * n
    groups_csv = getattr(a, "groups_csv", None)
    if groups_csv is not None:
        group_first = {}
        seen = set()
        with Path(groups_csv).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            if not {"path", "group"}.issubset(rows.fieldnames or []):
                raise ValueError("groups_csv must have path and group columns.")
            for line, row in enumerate(rows, start=2):
                raw_path = (row.get("path") or "").strip().replace("\\", "/")
                group = (row.get("group") or "").strip()
                if (not raw_path or not group or Path(raw_path).is_absolute()
                        or PureWindowsPath(raw_path).drive):
                    raise ValueError(f"groups_csv line {line}: need a relative path and nonempty group.")
                candidate = (train_root / raw_path).resolve()
                if candidate not in index_by_path:
                    raise ValueError(f"groups_csv line {line}: unknown training path {raw_path!r}.")
                i = index_by_path[candidate]
                if i in seen:
                    raise ValueError(f"groups_csv line {line}: duplicate training path {raw_path!r}.")
                seen.add(i)
                supplied_groups[i] = group
                if group in group_first:
                    union(i, group_first[group])
                else:
                    group_first[group] = i
        if len(seen) != n:
            missing = [relative_paths[i] for i in range(n) if i not in seen]
            raise ValueError(f"groups_csv is missing {len(missing)} training paths; examples: {missing[:5]}")

    roots = [root(i) for i in range(n)]
    group_lookup = {value: i for i, value in enumerate(sorted(set(roots)))}
    groups = np.array([group_lookup[value] for value in roots], dtype=np.int64)
    unique_groups = len(group_lookup)
    grouped = unique_groups < n
    for label, name in enumerate(classes):
        count = len(np.unique(groups[labels == label]))
        if count < a.folds:
            raise ValueError(f"Class {name!r} has only {count} independent groups for {a.folds} folds. "
                             "Reduce --folds or add independent labeled images.")
    dummy = np.zeros((n, 1), dtype=np.uint8)
    if grouped:
        splitter = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        splits = list(splitter.split(dummy, labels, groups))
    else:
        splitter = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
        splits = list(splitter.split(dummy, labels))
    fold_ids = np.full(n, -1, dtype=np.int64)
    fold_reports = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        for split_name, idx in (("train", train_idx), ("validation", val_idx)):
            missing = expected_labels - set(labels[idx].tolist())
            if missing:
                names = [classes[label] for label in sorted(missing)]
                raise ValueError(f"Fold {fold} {split_name} is missing classes {names}. "
                                 "Reduce --folds or revise plant grouping; no training was started.")
        if set(groups[train_idx]) & set(groups[val_idx]):
            raise RuntimeError(f"Internal split error: a group crosses fold {fold}.")
        fold_ids[val_idx] = fold
        fold_reports.append({
            "fold": fold,
            "train_count": int(len(train_idx)), "val_count": int(len(val_idx)),
            "train_class_counts": {str(name): int(np.sum(labels[train_idx] == i))
                                   for i, name in enumerate(classes)},
            "val_class_counts": {str(name): int(np.sum(labels[val_idx] == i))
                                 for i, name in enumerate(classes)}})
    if np.any(fold_ids < 0):
        raise RuntimeError("Internal split error: some images have no validation fold.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "label", "class", "fold", "group", "rgb_sha256", "supplied_group"])
        writer.writerows((relative_paths[i], int(labels[i]), classes[labels[i]], int(fold_ids[i]),
                          int(groups[i]), digests[i], supplied_groups[i]) for i in range(n))
    split_report = {
        "strategy": type(splitter).__name__, "seed": int(a.seed), "folds": int(a.folds),
        "image_count": n, "exact_duplicate_count": n - len(digest_first),
        "unique_groups": unique_groups, "groups_csv": str(groups_csv) if groups_csv else None,
        "limitation": "RGB hashes identify exact pixel duplicates only. Provide group metadata "
                      "for near duplicates and multiple photos of the same plant.",
        "fold_details": fold_reports}
    print(f"Splits: {split_report['strategy']}; {unique_groups} independent groups; "
          f"{split_report['exact_duplicate_count']} extra exact duplicates.", flush=True)
    return splits, groups if grouped else None, split_report
