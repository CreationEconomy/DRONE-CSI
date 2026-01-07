import argparse
import threading
import time
from collections import Counter
from typing import Optional

import numpy as np
import serial

from csi_utils import MAC_RE, parse_csi_line, enforce_fixed_length


def _normalize_mac(m: Optional[str]) -> Optional[str]:
    if not m:
        return None
    m = m.strip().upper()
    return m if MAC_RE.fullmatch(m) else None


def main():
    ap = argparse.ArgumentParser(description="ESP32 CSI를 라벨과 함께 수집하여 .npz로 저장합니다.")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="Serial port (예: /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=921600, help="Baud rate (기본 921600)")
    ap.add_argument("--target-mac", default=None, help="특정 MAC만 수집(예: AA:BB:CC:DD:EE:FF). 비우면 전체.")
    ap.add_argument("--out", default="csi_labeled.npz", help="출력 파일(.npz)")
    ap.add_argument("--max-seconds", type=float, default=0, help="0이면 무제한, 아니면 지정 시간 후 종료")
    args = ap.parse_args()

    target_mac = _normalize_mac(args.target_mac)
    mac_counter = Counter()

    current_label = {"value": "unknown"}
    stop_flag = {"value": False}

    def input_loop():
        print("라벨을 입력하세요. 예: near / far / wall / clear ...")
        print("종료: q (또는 quit/exit)")
        while not stop_flag["value"]:
            try:
                s = input("> ").strip()
            except EOFError:
                stop_flag["value"] = True
                break
            if not s:
                continue
            if s.lower() in ("q", "quit", "exit"):
                stop_flag["value"] = True
                break
            current_label["value"] = s
            print(f"[LABEL] {current_label['value']}")

    th = threading.Thread(target=input_loop, daemon=True)
    th.start()

    ser = serial.Serial(args.port, args.baud, timeout=0.2)
    ser.reset_input_buffer()
    print(f"Connected: {args.port} @ {args.baud}. TARGET_MAC={target_mac or 'ANY'}")

    ts_list = []
    rssi_list = []
    mac_list = []
    label_list = []
    csi_list = []
    rawline_list = []  # 디버깅용(원하면 저장)

    target_len = None
    skipped_len = 0
    total = 0

    t0 = time.time()
    try:
        while True:
            if stop_flag["value"]:
                break
            if args.max_seconds and (time.time() - t0) >= args.max_seconds:
                break

            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            parsed = parse_csi_line(line)
            if not parsed:
                continue

            if target_mac is not None and parsed.mac != target_mac:
                continue

            # 어떤 MAC이 들어오는지 관찰용(필터 없을 때 유용)
            if parsed.mac:
                mac_counter[parsed.mac] += 1

            if target_len is None:
                target_len = int(parsed.csi_raw.shape[0])

            fixed = enforce_fixed_length(parsed.csi_raw, target_len)
            if fixed is None:
                skipped_len += 1
                continue

            now = time.time()
            ts_list.append(now)
            rssi_list.append(parsed.rssi if parsed.rssi is not None else 127)
            mac_list.append(parsed.mac or "")
            label_list.append(current_label["value"])
            csi_list.append(fixed.astype(np.int16, copy=False))
            rawline_list.append(parsed.raw_line)

            total += 1
            if total % 200 == 0:
                print(
                    f"[{total}] label={current_label['value']} rssi={parsed.rssi if parsed.rssi is not None else 'N/A'} "
                    f"len={target_len} skipped_len={skipped_len}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()

    if total == 0:
        print("수집된 CSI 패킷이 없습니다. (트래픽/필터/MAC/포트 설정 확인)")
        return

    ts = np.asarray(ts_list, dtype=np.float64)
    rssi = np.asarray(rssi_list, dtype=np.int16)
    mac = np.asarray(mac_list, dtype="<U17")
    label = np.asarray(label_list, dtype="<U32")
    csi = np.stack(csi_list, axis=0).astype(np.int16, copy=False)  # (N, M)

    # raw line은 용량이 커질 수 있어 옵션화가 좋지만, 현재는 같이 저장(필요 없으면 지우면 됨)
    raw_line = np.asarray(rawline_list, dtype=object)

    np.savez_compressed(
        args.out,
        ts=ts,
        rssi=rssi,
        mac=mac,
        label=label,
        csi=csi,
        raw_line=raw_line,
        meta=np.array(
            {
                "port": args.port,
                "baud": int(args.baud),
                "target_mac": target_mac or "",
                "target_len": int(target_len),
                "skipped_len": int(skipped_len),
                "created_at": time.time(),
            },
            dtype=object,
        ),
    )

    print(f"저장 완료: {args.out}")
    print(f"- packets: {total}")
    print(f"- csi_raw_len: {target_len}")
    print(f"- skipped_len_mismatch: {skipped_len}")
    if target_mac is None and mac_counter:
        top = mac_counter.most_common(5)
        print(f"- top MACs (no filter): {top}")


if __name__ == "__main__":
    main()

