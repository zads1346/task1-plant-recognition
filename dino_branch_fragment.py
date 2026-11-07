"""Paste these helpers above main() in plant_study.py.

Uses existing np, torch, Path, json, seed_all(), get_loader(), eval_tf().
run_dino_branch returns (oof_probabilities, test_probabilities, JSON-safe report).
No labels or fold-dependent preprocessing are used while extracting features.
"""


def _dino_aligned_proba(classifier, features, n_classes):
    """Align sklearn probability columns with the ImageFolder class IDs."""
    probabilities = np.zeros((len(features), n_classes), dtype=np.float32)
    class_ids = np.asarray(classifier.classes_, dtype=int)
    if len(class_ids) != n_classes or not np.array_equal(
            np.sort(class_ids), np.arange(n_classes)):
        raise ValueError("DINO classifier training fold does not contain every class")
    probabilities[:, class_ids] = classifier.predict_proba(features)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("DINO classifier produced non-finite probabilities")
    return probabilities


def _dino_score(y, probabilities, metric, n_classes):
    from sklearn.metrics import f1_score
    predictions = probabilities.argmax(axis=1)
    if metric == "accuracy":
        return float(np.mean(predictions == y))
    return float(f1_score(y, predictions, labels=np.arange(n_classes),
                          average="macro", zero_division=0))


def _dino_inner_splits(y, groups, requested_folds, seed, n_classes):
    """Return indices local to the outer training set, or an explicit skip."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    min_count = int(np.bincount(y, minlength=n_classes).min())
    if groups is None:
        possible = min(requested_folds, min_count)
    else:
        independent_per_class = [len(np.unique(groups[y == label]))
                                 for label in range(n_classes)]
        possible = min(requested_folds, min(independent_per_class))
    # Some group layouts cannot retain every class in all training folds.
    # Decrease fold count, never fall back to splitting members of a group.
    for n_folds in range(possible, 1, -1):
        if groups is None:
            splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
            inner = list(splitter.split(np.zeros(len(y)), y))
        else:
            splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
            inner = list(splitter.split(np.zeros(len(y)), y, groups))
        complete = all(len(tr) and len(va)
                       and len(np.unique(y[tr])) == n_classes for tr, va in inner)
        disjoint = groups is None or all(
            not np.intersect1d(groups[tr], groups[va]).size for tr, va in inner)
        if complete and disjoint:
            return inner, None
    return [], ("Insufficient independent examples/groups for inner CV with "
                "all classes in each training fold; used a prespecified C.")


def _dino_new_classifier(c_value, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(c_value), solver="lbfgs", max_iter=2000,
                           random_state=seed),
    )


def _dino_fit_classifiers(features, y, test_features, splits, classes, *,
                          groups=None, c_values=(.01, .1, 1., 10.),
                          inner_folds=3, metric="accuracy", seed=42,
                          model_dir=None, model_name="dinov2_vits14", metadata=None):
    """Nested tuning on fixed frozen features; intentionally independently testable."""
    import time
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    import joblib

    n_classes = len(classes)
    y = np.asarray(y, dtype=int)
    groups = None if groups is None else np.asarray(groups)
    c_values = tuple(sorted(set(float(c) for c in c_values)))
    if not c_values or any(c <= 0 or not np.isfinite(c) for c in c_values):
        raise ValueError("DINO logistic C values must be finite and positive")
    if inner_folds < 2:
        raise ValueError("DINO inner folds must be >= 2")
    oof = np.zeros((len(y), n_classes), dtype=np.float32)
    test_mean = np.zeros((len(test_features), n_classes), dtype=np.float32)
    seen = np.zeros(len(y), dtype=int)
    fold_reports = []
    for fold, (tr, va) in enumerate(splits, 1):
        started = time.perf_counter()
        train_groups = None if groups is None else groups[tr]
        inner, skip_reason = _dino_inner_splits(
            y[tr], train_groups, inner_folds, seed + fold, n_classes)
        tuning = []
        for c_value in c_values if inner else ():
            scores, convergence_warnings = [], 0
            for inner_train, inner_valid in inner:
                # Pipeline.fit ensures the scaler never observes validation rows.
                candidate = _dino_new_classifier(c_value, seed + fold)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    candidate.fit(features[tr[inner_train]], y[tr[inner_train]])
                convergence_warnings += sum(issubclass(w.category, ConvergenceWarning)
                                            for w in caught)
                probabilities = _dino_aligned_proba(
                    candidate, features[tr[inner_valid]], n_classes)
                scores.append(_dino_score(y[tr[inner_valid]], probabilities,
                                          metric, n_classes))
            tuning.append({"C": c_value, "mean_score": float(np.mean(scores)),
                           "std_score": float(np.std(scores)), "fold_scores": scores,
                           "convergence_warnings": convergence_warnings})
        if tuning:
            chosen_c = max(tuning, key=lambda row: (row["mean_score"], -row["C"]))["C"]
        else:
            # Fixed fallback independent of outer validation labels.
            chosen_c = min(c_values, key=lambda c: (abs(np.log10(c)), c))
        classifier = _dino_new_classifier(chosen_c, seed + fold)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(features[tr], y[tr])
        convergence_warnings = sum(issubclass(w.category, ConvergenceWarning)
                                   for w in caught)
        valid_probabilities = _dino_aligned_proba(classifier, features[va], n_classes)
        oof[va] = valid_probabilities
        seen[va] += 1
        test_mean += _dino_aligned_proba(classifier, test_features, n_classes) / len(splits)
        checkpoint = None
        if model_dir is not None:
            checkpoint = Path(model_dir) / f"{model_name}_linear_fold{fold}.joblib"
            joblib.dump({"pipeline": classifier, "classes": list(classes),
                         "class_ids": list(range(n_classes)), "fold": fold,
                         "chosen_C": chosen_c, "backbone": model_name,
                         "metadata": metadata or {}}, checkpoint)
        item = {"fold": fold, "chosen_C": chosen_c, "inner_folds": len(inner),
                "tuning_status": "complete" if inner else "skipped",
                "tuning_skip_reason": skip_reason, "C_search": tuning,
                "train_count": len(tr), "valid_count": len(va),
                "accuracy": _dino_score(y[va], valid_probabilities, "accuracy", n_classes),
                "macro_f1": _dino_score(y[va], valid_probabilities, "macro_f1", n_classes),
                "convergence_warnings": convergence_warnings,
                "seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint) if checkpoint is not None else None}
        fold_reports.append(item)
        print(f"DINO linear fold {fold}/{len(splits)} C={chosen_c:g} "
              f"{metric}={item[metric]:.5f} ({item['seconds']:.1f}s)", flush=True)
        if skip_reason:
            print("DINO tuning note:", skip_reason, flush=True)
        if convergence_warnings or any(r["convergence_warnings"] for r in tuning):
            print("DINO logistic convergence warning; details saved in dino_analysis.json", flush=True)
    if not np.all(seen == 1):
        raise ValueError("DINO outer validation folds must cover each training row exactly once")
    report = {"model": model_name, "feature_extractor": "frozen pretrained DINOv2",
              "classifier": "StandardScaler + LogisticRegression (lbfgs)",
              "C_values": list(c_values), "metric": metric, "inner_folds_requested": inner_folds,
              "group_aware": groups is not None, "folds": fold_reports,
              "oof_accuracy": _dino_score(y, oof, "accuracy", n_classes),
              "oof_macro_f1": _dino_score(y, oof, "macro_f1", n_classes),
              "note": "Scaler and C tuning use only each outer training fold. "
                      "Frozen feature extraction observes no task labels. "
                      "Test probabilities are the mean of all outer-fold classifiers."}
    return oof, test_mean, report


def run_dino_branch(a, paths, y, test_paths, splits, classes, device, groups=None):
    """Extract once in FP32, then fit fold-specific linear classification heads.

    Call before CNN training so unavailable DINO weights fail immediately.
    Optional args: dino_model, dino_size, dino_batch_size, dino_inner_folds,
    dino_c_values. Defaults are ViT-S/14, 336, 32, 3, (.01,.1,1.,10.).
    """
    import time

    model_name = getattr(a, "dino_model", "dinov2_vits14")
    size = int(getattr(a, "dino_size", 336))
    batch = int(getattr(a, "dino_batch_size", 32))
    inner_folds = int(getattr(a, "dino_inner_folds", 3))
    c_values = getattr(a, "dino_c_values", (.01, .1, 1., 10.))
    if isinstance(c_values, str):
        c_values = tuple(float(c.strip()) for c in c_values.split(",") if c.strip())
    if size < 14 or size % 14 or batch < 1:
        raise ValueError("DINO size must be a positive multiple of 14 and batch >= 1")
    if model_name not in {"dinov2_vits14", "dinov2_vitb14"}:
        raise ValueError("Supported DINO backbones: dinov2_vits14, dinov2_vitb14")
    y = np.asarray(y, dtype=int)
    groups = None if groups is None else np.asarray(groups)
    n_classes = len(classes)
    if (len(paths) != len(y) or not len(test_paths)
            or not np.array_equal(np.unique(y), np.arange(n_classes))):
        raise ValueError("DINO paths/labels/classes are inconsistent or test set is empty")
    if groups is not None and len(groups) != len(y):
        raise ValueError("DINO groups must have one group ID per training image")
    splits = [(np.asarray(tr, dtype=int), np.asarray(va, dtype=int)) for tr, va in splits]
    seen = np.zeros(len(y), dtype=int)
    if len(splits) < 2:
        raise ValueError("DINO needs at least two outer folds")
    for tr, va in splits:
        if (not len(tr) or not len(va) or np.min(tr) < 0 or np.min(va) < 0
                or np.max(tr) >= len(y) or np.max(va) >= len(y)
                or len(np.unique(tr)) != len(tr) or len(np.unique(va)) != len(va)
                or np.intersect1d(tr, va).size
                or len(np.unique(y[tr])) != n_classes):
            raise ValueError("DINO outer split has invalid indices or a missing training class")
        if groups is not None and np.intersect1d(groups[tr], groups[va]).size:
            raise ValueError("DINO outer split leaks a group between training and validation")
        seen[va] += 1
    if not np.all(seen == 1):
        raise ValueError("DINO outer folds must validate every image exactly once")

    seed_all(a.seed)
    model_dir = Path(a.out) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading official pretrained {model_name}; Internet or a populated Torch Hub cache is required.",
          flush=True)
    try:
        # Official interface: https://github.com/facebookresearch/dinov2
        model = torch.hub.load("facebookresearch/dinov2", model_name,
                               pretrained=True, trust_repo=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load official pretrained {model_name}. Enable Kaggle Internet/Colab network "
            "or provide a populated Torch Hub cache containing the official repository and weights. "
            "No random-weight replacement was used. Disable the DINO branch explicitly to run CNN only."
        ) from exc
    model = model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    resize_mode = getattr(a, "resize_mode", "pad")
    transform = eval_tf(size, False, resize_mode=resize_mode)
    extraction_start = time.perf_counter()

    def extract(image_paths, title):
        loader = get_loader(image_paths, np.zeros(len(image_paths), dtype=int),
                            transform, batch, False, a.workers, device)
        chunks = []
        count = 0
        with torch.inference_mode():
            for step, (xb, _) in enumerate(loader, 1):
                # FP32 by default; no autocast or training-mode updates.
                with torch.autocast(device_type=device.type, enabled=False):
                    features = model(xb.to(device=device, dtype=torch.float32, non_blocking=True))
                if features.ndim != 2 or not torch.isfinite(features).all().item():
                    raise FloatingPointError("DINO returned invalid frozen features")
                chunks.append(features.cpu().numpy())
                count += len(xb)
                if step == 1 or step % 20 == 0 or step == len(loader):
                    print(f"DINO {title} features {count}/{len(image_paths)}", flush=True)
        return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)

    try:
        features = extract(paths, "train")
        test_features = extract(test_paths, "test")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    extraction_seconds = time.perf_counter() - extraction_start
    metadata = {"repository": "facebookresearch/dinov2", "pretrained": True,
                "image_size": size, "resize_mode": resize_mode, "precision": "float32",
                "feature_dimension": int(features.shape[1]), "cache": "none",
                "extraction_seconds": extraction_seconds}
    oof, test_mean, report = _dino_fit_classifiers(
        features, y, test_features, splits, classes, groups=groups, c_values=c_values,
        inner_folds=inner_folds, metric=a.metric, seed=a.seed, model_dir=model_dir,
        model_name=model_name, metadata=metadata)
    report.update(metadata)
    (Path(a.out) / "dino_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return oof, test_mean, report
