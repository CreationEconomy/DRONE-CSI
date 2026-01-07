import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import re
import threading
import queue
import time

# --- 설정 구간 ---
SERIAL_PORT = '/dev/ttyUSB1'
BAUD_RATE = 921600
# 시각화 모드:
# - "raw": 수신한 정수 배열 그대로(= I/Q가 번갈아 들어온 경우 파형이 요동치는 게 정상)
# - "amp_db_rel": I/Q → 복소 CSI → |H|를 패킷 내 최대값으로 정규화한 dB (가장 해석하기 쉬움)
PLOT_MODE = "amp_db_rel"
EPS = 1e-6
# 특정 MAC만 보고 싶으면 여기에 입력 (예: 드론 AP/BSSID).
# 디버깅/초기 확인 단계에서는 None으로 두고 전체 CSI_DATA를 먼저 확인하는 것을 권장.
TARGET_MAC = None  # 예: "AA:BB:CC:DD:EE:FF"
UPDATE_INTERVAL = 30  # ms (약 33 FPS)

csi_queue = queue.Queue(maxsize=1)
ser = None
running = True
last_data_time = 0
last_mac = None
last_rssi = None
axes_initialized = False

def _safe_int(s: str):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def _csi_to_xy(csi_data: np.ndarray):
    """
    esp32-csi-tool 출력의 CSI 배열은 보통 [I0 Q0 I1 Q1 ...] 형태(정수)입니다.
    이를 서브캐리어별 복소 CSI로 변환 후, 보기 쉬운 형태(진폭/위상 등)로 리턴합니다.
    """
    if csi_data is None or len(csi_data) == 0:
        return np.array([]), np.array([])

    if PLOT_MODE == "raw" or (len(csi_data) % 2 != 0):
        y = csi_data.astype(np.float32)
        x = np.arange(len(y))
        return x, y

    i = csi_data[0::2].astype(np.float32)
    q = csi_data[1::2].astype(np.float32)
    csi_complex = i + 1j * q
    amp = np.abs(csi_complex)

    if PLOT_MODE == "amp_db_rel":
        amp_max = float(np.max(amp)) if len(amp) else 1.0
        y = 20.0 * np.log10((amp + EPS) / (amp_max + EPS))
    else:
        # 기본은 진폭(선형)로 fallback
        y = amp

    x = np.arange(len(y))  # 서브캐리어 인덱스
    return x, y

def serial_reader():
    global running, ser, last_data_time, last_mac, last_rssi
    while running:
        try:
            if ser and ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # 1) CSI 데이터인지 확인
                # 2) (선택) TARGET_MAC으로 필터링
                if "CSI_DATA" in line:
                    parts = line.split(',')
                    mac = parts[2].strip() if len(parts) > 2 else None
                    if mac:
                        last_mac = mac
                    # 관례적으로 parts[3]에 RSSI가 들어오는 경우가 많음(펌웨어/포맷에 따라 다를 수 있음)
                    if len(parts) > 3:
                        last_rssi = _safe_int(parts[3])
                    if TARGET_MAC is not None and (not mac or mac.upper() != TARGET_MAC.upper()):
                        continue

                    csi_match = re.search(r'\[(.*?)\]', line)
                    if csi_match:
                        csi_raw = csi_match.group(1).split()
                        csi_data = np.array([int(x) for x in csi_raw])
                        last_data_time = time.time()
                        
                        if csi_queue.full():
                            try: csi_queue.get_nowait()
                            except: pass
                        csi_queue.put(csi_data)
        except:
            pass

def update(frame):
    global line, last_data_time, status_text, last_mac, last_rssi, axes_initialized
    # 5초간 해당 MAC 주소 데이터가 없으면 경고
    if last_data_time > 0 and time.time() - last_data_time > 5:
        if TARGET_MAC is None:
            status_text.set_text("⚠️ Searching for CSI_DATA (ANY MAC)...")
        else:
            status_text.set_text(f"⚠️ Searching for {TARGET_MAC}...")
        status_text.set_color('red')
        line.set_data([], [])
    else:
        if TARGET_MAC is None:
            status_text.set_text(f"● Tracking: ANY (last: {last_mac or 'N/A'}, RSSI: {last_rssi if last_rssi is not None else 'N/A'})")
        else:
            status_text.set_text(f"● Tracking: {TARGET_MAC} (RSSI: {last_rssi if last_rssi is not None else 'N/A'})")
        status_text.set_color('green')
        try:
            csi_data = csi_queue.get_nowait()
            x, y = _csi_to_xy(csi_data)
            line.set_data(x, y)

            # 첫 데이터 수신 시점에 축을 데이터에 맞춰 설정(패킷마다 변하지 않는 게 보통이므로 1회만)
            if not axes_initialized and len(x) > 0:
                ax = line.axes
                ax.set_xlim(0, len(x))
                if PLOT_MODE == "amp_db_rel":
                    ax.set_ylim(-60, 5)
                    ax.set_ylabel("Amplitude (dB, normalized per packet)")
                    ax.set_xlabel("Subcarrier index")
                else:
                    ax.set_ylim(float(np.min(y)) - 5, float(np.max(y)) + 5)
                axes_initialized = True
        except queue.Empty:
            pass
    return line, status_text

def main():
    global line, running, ser, status_text
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        ser.reset_input_buffer()
        print(f"Connected to {SERIAL_PORT}. Filtering MAC: {TARGET_MAC or 'ANY'}")
    except Exception as e:
        print(f"Error: {e}"); return

    threading.Thread(target=serial_reader, daemon=True).start()

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-', lw=1)
    status_text = ax.text(0.02, 0.95, "Waiting for data...", transform=ax.transAxes, color='orange')
    
    # 축은 첫 패킷 수신 시 자동으로 맞춥니다.
    ax.set_title(f"Real-time CSI ({PLOT_MODE}): {TARGET_MAC or 'ANY'}")
    ax.grid(True, alpha=0.3)
    
    ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=False)
    plt.show()
    running = False
    ser.close()

if __name__ == "__main__":
    main()