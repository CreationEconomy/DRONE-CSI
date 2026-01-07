import serial
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import re
from collections import deque
import warnings

# 한글 폰트 설정 (경고 제거)
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# --- 설정 ---
SERIAL_PORT = '/dev/ttyUSB0'  # 본인의 포트로 수정
BAUD_RATE = 921600            # menuconfig에서 설정한 속도
WINDOW_SIZE = 64              # 서브캐리어 개수 (20MHz 기준)
HISTORY_SIZE = 50             # 분석할 이전 데이터 개수
AVG_CHANGE_THRESHOLD = 0.001  # 평균 진폭 변화 임계값
VAR_CHANGE_THRESHOLD = 0.01  # 분산 변화 임계값

def parse_csi_data(line):
    """CSI 데이터를 파싱하여 진폭 계산"""
    match = re.search(r'\[(.*?)\]', line)
    if match:
        try:
            # 문자열을 쉼표로 분리하여 float 배열로 변환
            data_str = match.group(1).split(',')
            raw_data = []
            for x in data_str:
                x = x.strip()
                if x and x.replace('-', '', 1).replace('.', '', 1).isdigit():
                    raw_data.append(float(x))
            
            if len(raw_data) == 0:
                return None
                
            raw_data = np.array(raw_data)
            # 데이터 구조: [imag, real, imag, real, ...] 쌍으로 되어 있음
            # 진폭(Amplitude) 계산: sqrt(real^2 + imag^2)
            if len(raw_data) % 2 != 0:
                raw_data = raw_data[:-1]  # 홀수면 마지막 요소 제거
            
            if len(raw_data) == 0:
                return None
                
            imag = raw_data[::2]
            real = raw_data[1::2]
            amplitude = np.sqrt(real**2 + imag**2)
            return amplitude
        except (ValueError, IndexError) as e:
            # 파싱 실패 시 None 반환
            return None
    return None

def detect_wall_approach(avg_history, var_history):
    """CSI 데이터의 평균과 분산 변화를 분석하여 벽 접근 여부를 판단"""
    if len(avg_history) < 10:
        return False, 0.0
    
    # 최근 데이터와 이전 데이터 비교
    recent_avg = np.mean(list(avg_history)[-10:])
    old_avg = np.mean(list(avg_history)[:10])
    recent_var = np.mean(list(var_history)[-10:])
    old_var = np.mean(list(var_history)[:10])
    
    # 벽 접근 시 진폭 증가 및 분산 증가 경향
    avg_change = (recent_avg - old_avg) / (old_avg + 1e-6)
    var_change = (recent_var - old_var) / (old_var + 1e-6)
    
    # 벽 접근 신호: 진폭과 분산이 모두 증가
    is_approaching = (avg_change > AVG_CHANGE_THRESHOLD) and (var_change > VAR_CHANGE_THRESHOLD)
    confidence = min(abs(avg_change) + abs(var_change), 1.0)
    
    return is_approaching, confidence

# 데이터 이력 저장
avg_amplitude_history = deque(maxlen=HISTORY_SIZE)
variance_history = deque(maxlen=HISTORY_SIZE)
time_steps = deque(maxlen=HISTORY_SIZE)

# 그래프 설정 (3행 2열 레이아웃)
plt.ion()
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. CSI 진폭 (실시간)
ax1 = fig.add_subplot(gs[0, :])
line1, = ax1.plot(np.arange(WINDOW_SIZE), np.zeros(WINDOW_SIZE), 'b-', linewidth=2)
ax1.set_ylim(0, 50)
ax1.set_title("Real-time CSI Amplitude", fontsize=14, fontweight='bold')
ax1.set_xlabel("Subcarrier Index")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)

# 2. 평균 진폭 변화
ax2 = fig.add_subplot(gs[1, 0])
line2, = ax2.plot([], [], 'g-', linewidth=2)
ax2.set_title("Average Amplitude Trend", fontsize=12)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Mean Amplitude")
ax2.grid(True, alpha=0.3)

# 3. 진폭 분산 변화
ax3 = fig.add_subplot(gs[1, 1])
line3, = ax3.plot([], [], 'orange', linewidth=2)
ax3.set_title("Amplitude Variance", fontsize=12)
ax3.set_xlabel("Time Step")
ax3.set_ylabel("Variance")
ax3.grid(True, alpha=0.3)

# 4. 벽 접근 경고 (큰 텍스트 영역)
ax4 = fig.add_subplot(gs[2, :])
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
warning_text = ax4.text(0.5, 0.5, 'system waiting...', 
                        ha='center', va='center', 
                        fontsize=24, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

frame_count = 0

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}. Waiting for CSI data...")
    
    while True:
        if ser.in_waiting > 0:
            raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
            if "CSI_DATA" in raw_line:
                amp = parse_csi_data(raw_line)
                if amp is not None and len(amp) == WINDOW_SIZE:
                    # 통계 계산
                    mean_amp = np.mean(amp)
                    var_amp = np.var(amp)
                    
                    # 이력에 추가
                    avg_amplitude_history.append(mean_amp)
                    variance_history.append(var_amp)
                    time_steps.append(frame_count)
                    frame_count += 1
                    
                    # 1. CSI 진폭 업데이트
                    line1.set_ydata(amp)
                    ax1.set_ylim(0, max(50, np.max(amp) * 1.2))
                    
                    # 2. 평균 진폭 변화 업데이트
                    if len(time_steps) > 1:
                        line2.set_data(list(time_steps), list(avg_amplitude_history))
                        ax2.set_xlim(min(time_steps), max(time_steps))
                        ax2.set_ylim(min(avg_amplitude_history) * 0.9, 
                                     max(avg_amplitude_history) * 1.1)
                    
                    # 3. 분산 변화 업데이트
                    if len(time_steps) > 1:
                        line3.set_data(list(time_steps), list(variance_history))
                        ax3.set_xlim(min(time_steps), max(time_steps))
                        ax3.set_ylim(min(variance_history) * 0.9, 
                                     max(variance_history) * 1.1)
                    
                    # 4. 벽 접근 감지
                    is_approaching, confidence = detect_wall_approach(
                        avg_amplitude_history, variance_history)
                    
                    if is_approaching:
                        warning_text.set_text(f'⚠️ Wall Approaching! ⚠️\nConfidence: {confidence:.1%}')
                        warning_text.set_bbox(dict(boxstyle='round', 
                                                   facecolor='red', 
                                                   alpha=0.8))
                        warning_text.set_color('white')
                        # 상단 그래프 배경색도 변경
                        ax1.set_facecolor('#ffcccc')
                    else:
                        warning_text.set_text(f'✓ Safe Distance Maintained\nAverage Amplitude: {mean_amp:.2f}')
                        warning_text.set_bbox(dict(boxstyle='round', 
                                                   facecolor='lightgreen', 
                                                   alpha=0.8))
                        warning_text.set_color('darkgreen')
                        ax1.set_facecolor('white')
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
except KeyboardInterrupt:
    print("\nStopped by user")
finally:
    if 'ser' in locals():
        ser.close()
    print(f"Total frames processed: {frame_count}")