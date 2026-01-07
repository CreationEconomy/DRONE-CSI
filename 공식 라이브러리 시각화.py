import serial
import matplotlib.pyplot as plt
import numpy as np
import re

# --- 설정 ---
SERIAL_PORT = '/dev/ttyUSB0'  # 본인의 포트로 수정
BAUD_RATE = 921600            # menuconfig에서 설정한 속도
WINDOW_SIZE = 64              # 서브캐리어 개수 (20MHz 기준)

def parse_csi_data(line):
    # CSI_DATA...[...] 부분에서 숫자 배열만 추출
    match = re.search(r'\[(.*?)\]', line)
    if match:
        raw_data = np.fromstring(match.group(1), sep=',')
        # 데이터 구조: [imag, real, imag, real, ...] 쌍으로 되어 있음
        # 진폭(Amplitude) 계산: sqrt(real^2 + imag^2)
        # 데이터 개수가 짝수인지 확인 (쌍으로 이루어져야 함)
        if len(raw_data) % 2 != 0:
            raw_data = raw_data[:-1]  # 홀수면 마지막 요소 제거
        imag = raw_data[::2]
        real = raw_data[1::2]
        amplitude = np.sqrt(real**2 + imag**2)
        return amplitude
    return None

# 그래프 설정
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(np.arange(WINDOW_SIZE), np.zeros(WINDOW_SIZE))
ax.set_ylim(0, 50)  # 데이터 범위에 따라 조정 가능
ax.set_title("Real-time Drone CSI Amplitude")
ax.set_xlabel("Subcarrier Index")
ax.set_ylabel("Amplitude")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}. Waiting for CSI data...")
    
    while True:
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "CSI_DATA" in raw_line:
                amp = parse_csi_data(raw_line)
                if amp is not None and len(amp) == WINDOW_SIZE:
                    line.set_ydata(amp)
                    # 데이터에 맞춰 Y축 범위 자동 조절 (선택 사항)
                    # ax.set_ylim(0, np.max(amp) + 5) 
                    fig.canvas.draw()
                    fig.canvas.flush_events()
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    if 'ser' in locals():
        ser.close()