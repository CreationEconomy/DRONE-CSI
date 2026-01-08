import argparse
import ast
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from tscnn import TSCNNConfig, TimeSeries1DCNN, build_windows, iq128_to_amp64, save_tscnn_checkpoint


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_amp_frames_from_csv(csv_path: str) -> np.ndarray:
    """
    CSV의 data 컬럼(길이 128 I/Q raw)을 읽어, 진폭 64 프레임 시퀀스(T,64)로 변환합니다.
    손상된 패킷(길이 != 128)은 즉시 폐기합니다.
    """
    df = pd.read_csv(csv_path)
    if "data" not in df.columns:
        raise ValueError(f"'data' 컬럼이 없습니다: {csv_path}")

    total = int(len(df))
    ok_parse = 0
    ok_len = 0
    ok_amp = 0
    frames = []

    for cell in df["data"].astype(str).tolist():
        try:
            v = ast.literal_eval(cell)
        except Exception:
            continue
        ok_parse += 1

        if not isinstance(v, (list, tuple)) or len(v) != 128:
            continue
        ok_len += 1

        amp = iq128_to_amp64(v)
        if amp is None:
            continue
        ok_amp += 1
        frames.append(amp)

    if not frames:
        raise RuntimeError(
            f"유효 프레임이 없습니다. csv={csv_path}, total={total}, parsed={ok_parse}, len128={ok_len}, amp64={ok_amp}"
        )

    out = np.stack(frames, axis=0).astype(np.float32)  # (T,64)
    print(
        f"- {os.path.basename(csv_path)}: total={total}, parsed={ok_parse}, len128={ok_len}, amp64={ok_amp} -> frames={out.shape}"
    )
    return out


@torch.no_grad()
def eval_metrics(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    total = 0
    correct = 0
    tp_wall = 0
    fn_wall = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss_sum += float(loss.item()) * int(yb.shape[0])

        pred = torch.argmax(logits, dim=1)
        total += int(yb.shape[0])
        correct += int((pred == yb).sum().item())

        tp_wall += int(((yb == 1) & (pred == 1)).sum().item())
        fn_wall += int(((yb == 1) & (pred == 0)).sum().item())

    acc = (correct / total) if total else 0.0
    recall_wall = (tp_wall / (tp_wall + fn_wall)) if (tp_wall + fn_wall) else 0.0
    avg_loss = (loss_sum / total) if total else 0.0
    return avg_loss, acc, recall_wall


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone-CSI Time-Series 1D-CNN Trainer")
    parser.add_argument("--hover_csv", default="공중 호버.csv")
    parser.add_argument("--wall_csv", default="벽 가까이.csv")
    parser.add_argument("--out", default="csi_model_tscnn.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = TSCNNConfig()
    _set_seed(args.seed)
    device = torch.device(args.device)

    print("1) CSV 로딩 및 프레임(진폭 64) 변환...")
    hover_frames = load_amp_frames_from_csv(args.hover_csv)
    wall_frames = load_amp_frames_from_csv(args.wall_csv)

    print(f"2) 슬라이딩 윈도우 생성 (Window={cfg.window_size}, Stride(train)={cfg.stride_train})...")
    x_hover = build_windows(hover_frames, window_size=cfg.window_size, stride=cfg.stride_train)
    x_wall = build_windows(wall_frames, window_size=cfg.window_size, stride=cfg.stride_train)
    if x_hover.shape[0] == 0 or x_wall.shape[0] == 0:
        raise RuntimeError(f"윈도우가 부족합니다. hover_windows={x_hover.shape}, wall_windows={x_wall.shape}")

    y_hover = np.zeros((x_hover.shape[0],), dtype=np.int64)
    y_wall = np.ones((x_wall.shape[0],), dtype=np.int64)

    x = np.concatenate([x_hover, x_wall], axis=0)  # (N,64,20)
    y = np.concatenate([y_hover, y_wall], axis=0)  # (N,)

    print(f"- windows: hover={x_hover.shape[0]}, wall={x_wall.shape[0]}, total={x.shape[0]}")
    print(f"- model input shape: (N, C, L)=({x.shape[0]}, {x.shape[1]}, {x.shape[2]})")

    print("3) 시간순 분할 (Time-Series Split) - 데이터 누수 방지...")
    # 각 클래스별로 시간순 80/20 분할 후 합치기 (셔플 X)
    def time_split(data: np.ndarray, labels: np.ndarray, ratio: float = 0.8):
        n = int(data.shape[0])
        cut = int(n * ratio)
        return data[:cut], labels[:cut], data[cut:], labels[cut:]

    x_train_h, y_train_h, x_val_h, y_val_h = time_split(x_hover, y_hover, 0.8)
    x_train_w, y_train_w, x_val_w, y_val_w = time_split(x_wall, y_wall, 0.8)

    # 합친 후 train만 셔플 (val은 그대로)
    x_train = np.concatenate([x_train_h, x_train_w], axis=0)
    y_train = np.concatenate([y_train_h, y_train_w], axis=0)
    x_val = np.concatenate([x_val_h, x_val_w], axis=0)
    y_val = np.concatenate([y_val_h, y_val_w], axis=0)

    # train만 셔플
    rng = np.random.default_rng(args.seed)
    train_idx = rng.permutation(x_train.shape[0])
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    print(f"- train={x_train.shape[0]} (hover={x_train_h.shape[0]}, wall={x_train_w.shape[0]})")
    print(f"- val(test)={x_val.shape[0]} (hover={x_val_h.shape[0]}, wall={x_val_w.shape[0]})")

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("4) 모델 학습 (1D-CNN, Adam, CrossEntropy, EarlyStopping)...")
    model = TimeSeries1DCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * int(yb.shape[0])
            n_seen += int(yb.shape[0])

        train_loss = (running / n_seen) if n_seen else 0.0
        val_loss, val_acc, val_recall_wall = eval_metrics(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc*100:.2f}% | val_recall_wall={val_recall_wall*100:.2f}%"
        )

        if val_loss < (best_val_loss - 1e-6):
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            save_tscnn_checkpoint(
                args.out,
                model=model,
                config=cfg,
                extra={
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "val_acc": val_acc,
                    "val_recall_wall": val_recall_wall,
                },
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"EarlyStopping: val_loss가 {args.patience} epoch 동안 개선되지 않아 종료합니다. (best_epoch={best_epoch})")
                break

    print(f"모델 저장 완료: {args.out} (best_epoch={best_epoch}, best_val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    main()