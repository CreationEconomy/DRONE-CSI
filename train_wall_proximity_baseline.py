import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from csi_utils import csi_raw_to_complex, complex_to_amp_db_rel, majority_label


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    ts = data["ts"]
    rssi = data["rssi"]
    label = data["label"]
    csi = data["csi"]
    return ts, rssi, label, csi


def csi_matrix_to_amp_db_rel(csi_raw_2d: np.ndarray) -> np.ndarray:
    """
    (N, M=[I,Q interleaved]) → (N, S=subcarriers) amp_db_rel
    """
    n = csi_raw_2d.shape[0]
    out = []
    for i in range(n):
        cpx = csi_raw_to_complex(csi_raw_2d[i])
        if cpx is None:
            # 길이 불일치가 섞여 들어온 경우(원칙적으로 수집 단계에서 스킵됨)
            continue
        out.append(complex_to_amp_db_rel(cpx))
    return np.stack(out, axis=0).astype(np.float32, copy=False)


def window_features(
    amp_db: np.ndarray,
    rssi: np.ndarray,
    labels: np.ndarray,
    window_packets: int,
    step_packets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    amp_db: (N, S)
    rssi: (N,)
    labels: (N,)
    반환:
      X: (W, D)
      y: (W,) string labels
    """
    n = amp_db.shape[0]
    feats = []
    ys = []

    for start in range(0, n - window_packets + 1, step_packets):
        end = start + window_packets
        seg = amp_db[start:end]  # (w, S)
        seg_rssi = rssi[start:end].astype(np.float32)
        seg_labels = labels[start:end]

        y = majority_label(seg_labels)
        if y is None:
            continue

        mean_sc = np.mean(seg, axis=0)
        std_sc = np.std(seg, axis=0)
        mad_sc = np.mean(np.abs(np.diff(seg, axis=0)), axis=0) if window_packets >= 2 else np.zeros_like(mean_sc)

        rssi_mean = float(np.mean(seg_rssi[seg_rssi <= 0])) if np.any(seg_rssi <= 0) else float(np.mean(seg_rssi))
        rssi_std = float(np.std(seg_rssi[seg_rssi <= 0])) if np.any(seg_rssi <= 0) else float(np.std(seg_rssi))

        # feature vector
        f = np.concatenate([mean_sc, std_sc, mad_sc, np.array([rssi_mean, rssi_std], dtype=np.float32)], axis=0)
        feats.append(f)
        ys.append(y)

    if not feats:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype="<U1")

    X = np.stack(feats, axis=0).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype="<U32")
    return X, y


def main():
    ap = argparse.ArgumentParser(description="라벨된 CSI(.npz)로 벽 근접 분류 베이스라인을 학습합니다.")
    ap.add_argument("--inputs", nargs="+", required=True, help="입력 .npz 파일들")
    ap.add_argument("--window", type=int, default=80, help="윈도우 길이(패킷 수). 예: 80")
    ap.add_argument("--step", type=int, default=20, help="윈도우 스텝(패킷 수). 예: 20")
    ap.add_argument("--model", choices=["rf", "logreg"], default="rf", help="모델 종류")
    ap.add_argument("--out", default="wall_model.joblib", help="모델 저장 경로(joblib)")
    args = ap.parse_args()

    try:
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import joblib
    except Exception as e:
        print("scikit-learn/joblib이 필요합니다. 설치 예:")
        print("  pip install -U scikit-learn joblib")
        raise

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    groups: List[int] = []

    for gi, p in enumerate(args.inputs):
        ts, rssi, label, csi = load_npz(p)
        amp_db = csi_matrix_to_amp_db_rel(csi)
        # amp_db가 길이 불일치로 일부 drop될 수 있으니, rssi/label도 같은 길이로 맞춤
        n = amp_db.shape[0]
        rssi2 = np.asarray(rssi[:n])
        label2 = np.asarray(label[:n])

        X, y = window_features(amp_db, rssi2, label2, window_packets=args.window, step_packets=args.step)
        if X.size == 0:
            print(f"[SKIP] {p}: usable windows=0")
            continue

        Xs.append(X)
        ys.append(y)
        groups.extend([gi] * X.shape[0])

        print(f"[OK] {Path(p).name}: packets={n} windows={X.shape[0]} labels={sorted(set(y.tolist()))}")

    if not Xs:
        print("학습할 데이터가 없습니다. (라벨이 unknown 뿐이거나, 윈도우 설정이 너무 큼)")
        return

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    groups = np.asarray(groups, dtype=np.int32)

    # 파일 단위로 train/test 분리(한 파일이면 시간 순서 분리보다 낫지만, 파일이 1개면 그룹 분리 불가)
    if len(np.unique(groups)) >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(splitter.split(X_all, y_all, groups=groups))
    else:
        # fallback: 앞 70% train, 뒤 30% test
        n = X_all.shape[0]
        cut = int(n * 0.7)
        train_idx = np.arange(0, cut)
        test_idx = np.arange(cut, n)

    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    else:
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    print("\n=== Test Report ===")
    print(classification_report(y_te, y_pred, digits=4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_pred))

    joblib.dump(clf, args.out)
    print(f"\n모델 저장: {args.out}")


if __name__ == "__main__":
    main()

