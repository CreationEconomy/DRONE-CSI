#!/usr/bin/env python3
# -*-coding:utf-8-*-

import sys
import csv
import json
import argparse
import pandas as pd
import numpy as np
import joblib
import serial
import scipy.stats as stats  # 특징 추출용
from collections import deque, Counter
from io import StringIO

from PyQt5.Qt import *
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QThread

# --- 글로벌 변수 ---
latest_raw_data = None
# ------------------

# CSI 데이터 설정
CSI_DATA_INDEX = 200
CSI_DATA_COLUMNS = 490
DATA_COLUMNS_NAMES = ['type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth', 'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding', 'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel', 'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word', 'data']

# 그래프용 버퍼
csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)

# ==========================================
# [핵심] 특징 추출 함수 (학습 코드와 동일해야 함)
# ==========================================
def extract_features(csi_list):
    if len(csi_list) == 0: return [0]*9
    data = np.array(csi_list)
    
    mean_val = np.mean(data)
    std_val = np.std(data)
    max_val = np.max(data)
    min_val = np.min(data)
    range_val = max_val - min_val
    skew_val = stats.skew(data)
    kurt_val = stats.kurtosis(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    energy = np.sum(data ** 2) / len(data)
    
    return [mean_val, std_val, max_val, min_val, range_val, skew_val, kurt_val, iqr, energy]

class csi_data_graphical_window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced CSI Detector (Feature Engineering)")
        self.resize(1280, 900)

        # 레이아웃 설정
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # 1. 상태 표시 라벨 (크고 잘 보이게)
        self.status_label = QLabel("Loading Model...")
        self.status_label.setFixedHeight(80)
        self.status_label.setStyleSheet("background-color: #333; color: white; font-size: 32px; font-weight: bold; border: 4px solid gray; qproperty-alignment: AlignCenter;")
        main_layout.addWidget(self.status_label)

        # 2. 그래프 영역
        graph_layout = QGridLayout()
        main_layout.addLayout(graph_layout)

        # 모델 로드
        try:
            self.model = joblib.load('csi_model_advanced.pkl')
            self.model_loaded = True
            self.status_label.setText("READY: Waiting for Signal...")
            print("모델 로드 성공: csi_model_advanced.pkl")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            self.status_label.setText("Model Error!")
            self.model_loaded = False
        
        # [안정화] 예측값 버퍼 (최근 15개 투표)
        self.prediction_buffer = deque(maxlen=15)

        # 그래프 위젯들
        self.plot_phase = PlotWidget(title="Phase (Last Frame)")
        self.plot_phase.setYRange(-7, 7)
        graph_layout.addWidget(self.plot_phase, 0, 0)

        self.plot_iq = PlotWidget(title="IQ Plot")
        self.plot_iq.setAspectLocked(True)
        self.plot_iq.setRange(xRange=(-30, 30), yRange=(-30, 30))
        self.iq_scatter = ScatterPlotItem(size=7)
        self.plot_iq.addItem(self.iq_scatter)
        graph_layout.addWidget(self.plot_iq, 0, 1)

        self.plot_amp = PlotWidget(title="Amplitude History")
        graph_layout.addWidget(self.plot_amp, 1, 0, 1, 2)

        # 데이터 연결
        self.curve_phase = self.plot_phase.plot([], pen='y')
        self.curves_amp = []
        for i in range(CSI_DATA_COLUMNS):
            self.curves_amp.append(self.plot_amp.plot([], pen=(255,255,255, 30))) # 투명도 적용

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50) # 20 FPS
        
        self.iq_colors = []

    def update_colors(self, colors):
        self.iq_colors = colors

    def update_ui(self):
        # ==========================================
        # [핵심] 실시간 예측 로직
        # ==========================================
        global latest_raw_data
        
        if self.model_loaded and latest_raw_data is not None:
            raw_list = latest_raw_data
            
            # 128 서브캐리어인 경우만 처리
            if len(raw_list) == 128:
                try:
                    # 1. 특징 추출 (9개 값)
                    features = extract_features(raw_list)
                    feature_vector = np.array(features).reshape(1, -1)
                    
                    # 2. 모델 예측
                    pred = self.model.predict(feature_vector)[0]
                    self.prediction_buffer.append(pred)
                    
                    # 3. 다수결 투표 (Smoothing)
                    if len(self.prediction_buffer) >= 5:
                        counts = Counter(self.prediction_buffer)
                        final_decision = counts.most_common(1)[0][0]
                        confidence = counts[final_decision] / len(self.prediction_buffer) * 100
                        
                        if final_decision == 0:
                            self.status_label.setText(f"🛸 HOVER (공중)  [{confidence:.0f}%]")
                            self.status_label.setStyleSheet("background-color: #008800; color: white; font-size: 32px; font-weight: bold; border: 4px solid white;")
                        else:
                            self.status_label.setText(f"🧱 WALL (벽 감지!)  [{confidence:.0f}%]")
                            self.status_label.setStyleSheet("background-color: #DD0000; color: white; font-size: 32px; font-weight: bold; border: 4px solid yellow;")

                except Exception as e:
                    print(f"Pred Error: {e}")

        # ==========================================
        # 그래프 업데이트 (시각화)
        # ==========================================
        # Phase
        last_phase = np.angle(csi_data_complex[-1])
        # 유효한 서브캐리어만 그림 (128개라고 가정)
        valid_len = len(latest_raw_data) if latest_raw_data else 0
        if valid_len > 0:
            self.curve_phase.setData(last_phase[:valid_len])

        # IQ Plot
        i_val = np.real(csi_data_complex[-1])
        q_val = np.imag(csi_data_complex[-1])
        points = []
        for idx in range(valid_len):
            c = self.iq_colors[idx] if idx < len(self.iq_colors) else (200,200,200)
            points.append({'pos': (i_val[idx], q_val[idx]), 'brush': pg.mkBrush(c)})
        self.iq_scatter.setData(points)

        # Amplitude (일부만 그림 - 성능 최적화)
        # 10개 간격으로 몇 개만 그려서 전체 추이 확인
        amp_data = np.abs(csi_data_complex)
        for i in range(0, valid_len, 10): 
            if i < len(self.curves_amp):
                self.curves_amp[i].setData(amp_data[:, i])


# --- 시리얼 통신 스레드 ---
def csi_reader(port, csv_writer, log_file, callback_color):
    global latest_raw_data, csi_data_complex
    
    try:
        ser = serial.Serial(port, 921600, timeout=1)
        print("Serial Open Success")
    except:
        print("Serial Open Failed")
        return

    while True:
        try:
            line = ser.readline()
            if not line: continue
            
            try: text = line.decode('utf-8').strip()
            except: continue

            if "CSI_DATA" not in text: continue

            # 파싱
            csv_reader = csv.reader(StringIO(text))
            row = next(csv_reader)
            
            if len(row) < 25: continue
            
            try: 
                raw_data = json.loads(row[-1])
                latest_raw_data = raw_data # 실시간 데이터 갱신
            except: continue
            
            csi_len = int(row[-3])
            if csi_len != len(raw_data): continue

            # 버퍼 업데이트 (Shift)
            csi_data_complex[:-1] = csi_data_complex[1:]
            for i in range(csi_len // 2):
                csi_data_complex[-1][i] = complex(raw_data[i*2+1], raw_data[i*2])
            
            # 색상 콜백 (최초 1회 설정용)
            if csi_len == 128:
                # 128개 기준 색상표 생성
                cols = []
                for i in range(128):
                    if i < 32: cols.append((255,0,0))
                    elif i < 64: cols.append((0,255,0))
                    else: cols.append((0,0,255))
                callback_color.emit(cols)

            # 저장
            csv_writer.writerow(row)

        except Exception as e:
            # print(e)
            pass

class SerialThread(QThread):
    color_signal = pyqtSignal(object)
    
    def __init__(self, port, store, log):
        super().__init__()
        self.port = port
        self.store = store
        self.log = log
    
    def run(self):
        with open(self.store, 'w', newline='') as f1, open(self.log, 'w') as f2:
            writer = csv.writer(f1)
            writer.writerow(DATA_COLUMNS_NAMES)
            csi_reader(self.port, writer, f2, self.color_signal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', required=True)
    parser.add_argument('-s', '--store', default='csi_data.csv')
    parser.add_argument('-l', '--log', default='csi_log.txt')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    
    win = csi_data_graphical_window()
    t = SerialThread(args.port, args.store, args.log)
    t.color_signal.connect(win.update_colors)
    t.start()
    
    win.show()
    sys.exit(app.exec())