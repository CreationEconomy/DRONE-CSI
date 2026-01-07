import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import re
import threading
import queue
import time

# --- 설정 구간 ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 921600
# 휴대폰 핫스팟 MAC 주소
TARGET_MAC = "FA:29:AA:6F:BF:24" 
UPDATE_INTERVAL = 30  # ms (약 33 FPS)

csi_queue = queue.Queue(maxsize=1)
ser = None
running = True
last_data_time = 0

def serial_reader():
    global running, ser, last_data_time
    while running:
        try:
            if ser and ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                # 1. CSI 데이터인지 확인
                # 2. 지정한 TARGET_MAC이 포함되어 있는지 필터링
                if "CSI_DATA" in line and TARGET_MAC.upper() in line.upper():
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
    global line, last_data_time, status_text
    # 5초간 해당 MAC 주소 데이터가 없으면 경고
    if last_data_time > 0 and time.time() - last_data_time > 5:
        status_text.set_text(f"⚠️ Searching for {TARGET_MAC}...")
        status_text.set_color('red')
        line.set_data([], [])
    else:
        status_text.set_text(f"● Tracking: {TARGET_MAC}")
        status_text.set_color('green')
        try:
            csi_data = csi_queue.get_nowait()
            line.set_data(np.arange(len(csi_data)), csi_data)
        except queue.Empty:
            pass
    return line, status_text

def main():
    global line, running, ser, status_text
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        ser.reset_input_buffer()
        print(f"Connected to {SERIAL_PORT}. Filtering MAC: {TARGET_MAC}")
    except Exception as e:
        print(f"Error: {e}"); return

    threading.Thread(target=serial_reader, daemon=True).start()

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-', lw=1)
    status_text = ax.text(0.02, 0.95, "Waiting for data...", transform=ax.transAxes, color='orange')
    
    ax.set_ylim(-120, 50) # 진폭 범위 설정
    ax.set_xlim(0, 256)   # 서브캐리어 인덱스 (256개 기준)
    ax.set_title(f"Real-time CSI: {TARGET_MAC}")
    ax.grid(True, alpha=0.3)
    
    ani = FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=True)
    plt.show()
    running = False
    ser.close()

if __name__ == "__main__":
    main()