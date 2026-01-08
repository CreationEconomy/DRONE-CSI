#!/usr/bin/env python3
# -*-coding:utf-8-*-

import sys
import csv
import json
import argparse
import numpy as np
import serial
import torch
from collections import deque
from io import StringIO

from PyQt5.Qt import *
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal, QThread

from tscnn import TSCNNConfig, iq128_to_amp64, load_tscnn_checkpoint

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

class csi_data_graphical_window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drone-CSI-Sense (Time-Series 1D-CNN)")
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

        # Time-Series CNN 설정(지시서 스펙)
        self.cfg = TSCNNConfig()
        self.model_loaded = False  # SerialThread에서 로드 후 signal로 알려줌

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

    def on_model_status(self, payload: object):
        """
        SerialThread에서 모델 로드 성공/실패를 알려줍니다.
        payload 예:
          {"ok": True, "model_path": "...", "cfg": {...}}
          {"ok": False, "error": "..."}
        """
        try:
            p = dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            p = {}

        ok = bool(p.get("ok", False))
        if ok:
            self.model_loaded = True
            self.status_label.setText(f"READY: Buffer 0/{self.cfg.window_size}")
            self.status_label.setStyleSheet(
                "background-color: #333; color: white; font-size: 32px; font-weight: bold; border: 4px solid gray; qproperty-alignment: AlignCenter;"
            )
            print(f"모델 로드 성공: {p.get('model_path', '')}")
        else:
            self.model_loaded = False
            err = str(p.get("error", "unknown error"))
            self.status_label.setText(f"Model Error!\n{err}")
            self.status_label.setStyleSheet(
                "background-color: #550000; color: white; font-size: 20px; font-weight: bold; border: 4px solid yellow; qproperty-alignment: AlignCenter;"
            )
            print(f"모델 로드 실패: {err}")

    def on_prediction(self, payload: object):
        """
        SerialThread에서 매 패킷(Stride=1) 추론 결과를 전달합니다.
        payload 예:
          {"buffer_len": 7, "window_size": 20, "warming_up": True}
          {"buffer_len": 20, "window_size": 20, "pred": 1, "wall_prob": 0.93, "wall_warning": True}
        """
        if not self.model_loaded:
            return
        try:
            p = dict(payload) if isinstance(payload, dict) else {}
        except Exception:
            return

        buf = int(p.get("buffer_len", 0))
        win = int(p.get("window_size", self.cfg.window_size))
        if p.get("warming_up", False) or ("pred" not in p):
            self.status_label.setText(f"READY: Buffer {buf}/{win}")
            return

        wall_prob = float(p.get("wall_prob", 0.0))
        wall_warning = bool(p.get("wall_warning", False))
        wall_pct = wall_prob * 100.0

        if wall_warning:
            self.status_label.setText(f"🧱 벽 경고! (3연속)\nWall {wall_pct:.0f}%")
            self.status_label.setStyleSheet(
                "background-color: #DD0000; color: white; font-size: 32px; font-weight: bold; border: 4px solid yellow; qproperty-alignment: AlignCenter;"
            )
        else:
            self.status_label.setText(f"🛸 안전 호버링\nWall {wall_pct:.0f}%")
            self.status_label.setStyleSheet(
                "background-color: #008800; color: white; font-size: 32px; font-weight: bold; border: 4px solid white; qproperty-alignment: AlignCenter;"
            )

    def update_ui(self):
        # ==========================================
        # 그래프 업데이트 (시각화)
        # ==========================================
        global latest_raw_data
        # Phase
        last_phase = np.angle(csi_data_complex[-1])
        # 유효한 서브캐리어만 그림 (I/Q 128 -> complex 64)
        valid_len = 0
        if latest_raw_data:
            try:
                valid_len = int(len(latest_raw_data) // 2) if (len(latest_raw_data) % 2 == 0) else 0
            except Exception:
                valid_len = 0
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
def csi_reader(port, csv_writer, callback_color, pred_signal, model, cfg: TSCNNConfig, device: torch.device):
    global latest_raw_data, csi_data_complex
    frame_buffer = deque(maxlen=cfg.window_size)
    pred_streak = deque(maxlen=3)  # 최근 3번 예측 결과가 모두 Wall(1)일 때만 경고
    
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

            # ==========================================
            # [핵심] Time-Series 1D-CNN 실시간 추론 (Stride=1)
            # ==========================================
            if csi_len == cfg.iq_len:
                amp64 = iq128_to_amp64(raw_data)
                if amp64 is not None:
                    frame_buffer.append(amp64)

                    # 버퍼 워밍업 상태 전달
                    if len(frame_buffer) < cfg.window_size:
                        pred_signal.emit(
                            {
                                "warming_up": True,
                                "buffer_len": len(frame_buffer),
                                "window_size": cfg.window_size,
                            }
                        )
                    else:
                        # 입력 텐서: (1,64,20)
                        window = np.stack(frame_buffer, axis=0).T.astype(np.float32)
                        x = torch.from_numpy(window).unsqueeze(0).to(device).float()
                        with torch.no_grad():
                            logits = model(x)
                            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                            pred = int(np.argmax(probs))
                            wall_prob = float(probs[1])

                        pred_streak.append(pred)
                        wall_warning = (len(pred_streak) == 3) and all(p == 1 for p in pred_streak)

                        pred_signal.emit(
                            {
                                "buffer_len": len(frame_buffer),
                                "window_size": cfg.window_size,
                                "pred": pred,
                                "wall_prob": wall_prob,
                                "wall_warning": wall_warning,
                            }
                        )

            # 저장
            csv_writer.writerow(row)

        except Exception as e:
            # print(e)
            pass

class SerialThread(QThread):
    color_signal = pyqtSignal(object)
    model_signal = pyqtSignal(object)
    pred_signal = pyqtSignal(object)
    
    def __init__(self, port, store, log, model_path: str, device: str):
        super().__init__()
        self.port = port
        self.store = store
        self.log = log
        self.model_path = model_path
        self.device = device
    
    def run(self):
        # 모델 로드 (실패하면 UI에 알리고 종료)
        try:
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA를 요청했지만 torch.cuda.is_available()=False 입니다. --device cpu로 실행하세요.")
            model, cfg, _ = load_tscnn_checkpoint(self.model_path, device=self.device)
            device = torch.device(self.device)
            self.model_signal.emit({"ok": True, "model_path": self.model_path, "cfg": cfg.__dict__})
        except Exception as e:
            self.model_signal.emit({"ok": False, "model_path": self.model_path, "error": str(e)})
            return

        with open(self.store, 'w', newline='') as f1, open(self.log, 'w') as f2:
            writer = csv.writer(f1)
            writer.writerow(DATA_COLUMNS_NAMES)
            csi_reader(self.port, writer, self.color_signal, self.pred_signal, model, cfg, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', required=True)
    parser.add_argument('-s', '--store', default='csi_data.csv')
    parser.add_argument('-l', '--log', default='csi_log.txt')
    parser.add_argument('-m', '--model', default='csi_model_tscnn.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    app = QApplication(sys.argv)
    
    win = csi_data_graphical_window()
    t = SerialThread(args.port, args.store, args.log, args.model, args.device)
    t.color_signal.connect(win.update_colors)
    t.model_signal.connect(win.on_model_status)
    t.pred_signal.connect(win.on_prediction)
    t.start()
    
    win.show()
    sys.exit(app.exec())