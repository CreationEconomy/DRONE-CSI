#!/usr/bin/env python3
# -*-coding:utf-8-*-

# SPDX-FileCopyrightText: 2021-2025 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
#

# WARNING: we don't check for Python build-time dependencies until
# check_environment() function below. If possible, avoid importing
# any external libraries here - put in external script, or import in
# their specific function instead.

# sudo chmod 666 /dev/ttyUSB0
# python ./data.py -p /dev/ttyUSB0

import sys
import os
from datetime import datetime

# Qt 플러그인 충돌 방지 및 플랫폼 설정
# OpenCV의 내장 Qt 플러그인 대신 PyQt5의 플러그인을 사용하도록 강제 지정합니다.
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/local/lib64/python3.14/site-packages/PyQt5/Qt5/plugins"
# Fedora Wayland 환경에서 sudo 실행 시 발생할 수 있는 xcb 로드 오류를 해결 위해 xcb 플랫폼 강제 지정
os.environ["QT_QPA_PLATFORM"] = "xcb"
# X11 MIT-SHM extension 관련 오류 방지
os.environ["QT_X11_NO_MITSHM"] = "1"

import csv
import json
import argparse
import pandas as pd
import numpy as np

import serial
from os import path
from io import StringIO

from PyQt5.Qt import *
from pyqtgraph import PlotWidget
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton
from PyQt5.QtGui import QImage, QPixmap
import pyqtgraph as pg
from pyqtgraph import ScatterPlotItem
from PyQt5.QtCore import pyqtSignal, QThread, Qt, QTimer
import threading
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.api as sm

# DJI Tello SDK
from djitellopy import Tello
# import cv2  # 비디오 기능 제거로 불필요

# Reduce displayed waveforms to avoid display freezes
CSI_VAID_SUBCARRIER_INTERVAL = 1
csi_vaid_subcarrier_len =0

CSI_DATA_INDEX = 200  # buffer size
CSI_DATA_COLUMNS = 490
DATA_COLUMNS_NAMES_C5C6 = ['type', 'id', 'mac', 'rssi', 'rate','noise_floor','fft_gain','agc_gain', 'channel', 'local_timestamp',  'sig_len', 'rx_state', 'len', 'first_word', 'data']
DATA_COLUMNS_NAMES = ['type', 'id', 'mac', 'rssi', 'rate', 'sig_mode', 'mcs', 'bandwidth', 'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
                      'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'secondary_channel', 'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word', 'data']

# Tello 데이터 컬럼명 추가 (시스템 타임스탬프 포함)
TELLO_COLUMNS_NAMES = ['system_timestamp', 'tello_battery', 'tello_height', 'tello_temperature', 'tello_flight_time', 
                       'tello_barometer', 'tello_tof', 'tello_pitch', 'tello_roll', 'tello_yaw',
                       'tello_speed_x', 'tello_speed_y', 'tello_speed_z', 
                       'tello_accel_x', 'tello_accel_y', 'tello_accel_z']

# 전체 CSV 컬럼명 (CSI + Tello)
COMBINED_COLUMNS_NAMES = DATA_COLUMNS_NAMES + TELLO_COLUMNS_NAMES
COMBINED_COLUMNS_NAMES_C5C6 = DATA_COLUMNS_NAMES_C5C6 + TELLO_COLUMNS_NAMES

csi_data_array = np.zeros(
    [CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.float64)
csi_data_phase = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.float64)
csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
fft_gains = []
agc_gains = []

# Tello 상태 데이터 저장용 전역 변수
tello_state_data = {
    'battery': 0,
    'height': 0,
    'temperature': 0,
    'flight_time': 0,
    'barometer': 0.0,
    'tof': 0,
    'pitch': 0,
    'roll': 0,
    'yaw': 0,
    'speed_x': 0,
    'speed_y': 0,
    'speed_z': 0,
    'acceleration_x': 0.0,
    'acceleration_y': 0.0,
    'acceleration_z': 0.0,
}

# Tello 연결 상태
tello_connected = False
tello_instance = None


class TelloThread(QThread):
    """Tello 상태 데이터를 주기적으로 읽어오는 스레드"""
    state_updated = pyqtSignal(dict)
    frame_updated = pyqtSignal(np.ndarray)
    
    def __init__(self, tello):
        super().__init__()
        self.tello = tello
        self.running = True
        self._error_count = 0
        self._frame_counter = 0  # 비디오 프레임 카운터
        
    def run(self):
        global tello_state_data
        while self.running:
            try:
                # 성능 최적화: 한 번에 모든 상태 가져오기 (가능한 경우)
                state = {
                    'battery': self.tello.get_battery(),
                    'height': self.tello.get_height(),
                    'temperature': self.tello.get_temperature(),
                    'flight_time': self.tello.get_flight_time(),
                    'barometer': self.tello.get_barometer(),
                    'tof': self.tello.get_distance_tof(),
                    'pitch': self.tello.get_pitch(),
                    'roll': self.tello.get_roll(),
                    'yaw': self.tello.get_yaw(),
                    'speed_x': self.tello.get_speed_x(),
                    'speed_y': self.tello.get_speed_y(),
                    'speed_z': self.tello.get_speed_z(),
                    'acceleration_x': self.tello.get_acceleration_x(),
                    'acceleration_y': self.tello.get_acceleration_y(),
                    'acceleration_z': self.tello.get_acceleration_z(),
                }
                tello_state_data = state
                self.state_updated.emit(state)
                self._error_count = 0  # 성공 시 에러 카운트 초기화
                
                # 비디오 프레임 읽기 비활성화 (성능 최적화)
                # self._frame_counter += 1
                # if self._frame_counter >= 2:
                #     self._frame_counter = 0
                #     frame = self.tello.get_frame_read().frame
                #     if frame is not None:
                #         self.frame_updated.emit(frame)
                    
            except Exception as e:
                self._error_count += 1
                if self._error_count <= 5:  # 처음 5번만 로그 출력
                    print(f"Tello 데이터 읽기 오류: {e}")
                if self._error_count > 20:  # 20번 연속 실패 시 중단
                    print("Tello 연결 실패. 스레드 종료.")
                    self.running = False
                
            time.sleep(0.1)  # 10Hz로 감소 (성능 최적화)
            
    def stop(self):
        self.running = False


class TelloConnectThread(QThread):
    """Tello 드론 연결을 백그라운드에서 수행하는 스레드"""
    connection_result = pyqtSignal(object, str)  # (tello_instance or None, message)
    
    def __init__(self):
        super().__init__()
        
    def run(self):
        try:
            tello = Tello()
            tello.connect()
            battery = tello.get_battery()
            tello.streamon()
            self.connection_result.emit(tello, f"연결 성공! 배터리: {battery}%")
        except Exception as e:
            self.connection_result.emit(None, f"연결 실패: {str(e)}")


class csi_data_graphical_window(QWidget):
    def __init__(self, tello=None):
        super().__init__()
        self.tello = tello
        self.tello_flying = False
        self.tello_thread = None  # Tello 상태 수집 스레드
        self.tello_connect_thread = None  # Tello 연결 스레드
        
        # 키보드 조종을 위한 키 상태 추적
        self.pressed_keys = set()
        self.control_timer = QTimer()
        self.control_timer.timeout.connect(self.send_control_commands)
        self.control_timer.start(50)  # 50ms마다 명령 전송 (20Hz)

        self.setMinimumSize(1280, 720)
        self.resize(1920, 1000)
        
        # 다크 테마 배경
        self.setStyleSheet("background-color: #1a1a1a;")

        # ========== 메인 레이아웃 ==========
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # === 왼쪽: CSI 그래프들 (비율 7) ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)
        
        # 상단 그래프 영역 (Phase + IQ)
        top_graphs = QHBoxLayout()
        
        self.plotWidget_ted = PlotWidget()
        self.plotWidget_ted.setYRange(-2*np.pi, 2*np.pi)
        self.plotWidget_ted.addLegend()
        self.plotWidget_ted.setTitle('Phase Data - Last Frame')
        self.plotWidget_ted.setLabel('left', 'Phase (rad)')
        self.plotWidget_ted.setLabel('bottom', 'Subcarrier Index')

        self.csi_amplitude_array = np.abs(csi_data_complex)
        self.csi_phase_array = np.angle(csi_data_complex)
        self.curve = self.plotWidget_ted.plot([], name='CSI Row Data', pen='r')
        
        self.plotWidget_iq = PlotWidget()
        self.plotWidget_iq.setLabel('left', 'Q (Imag)')
        self.plotWidget_iq.setLabel('bottom', 'I (Real)')
        self.plotWidget_iq.setTitle('IQ Plot - Last Frame')
        view_box = self.plotWidget_iq.getViewBox()
        view_box.setRange(QtCore.QRectF(-30, -30, 60, 60))
        self.plotWidget_iq.getViewBox().setAspectLocked(True)
        self.iq_scatter = ScatterPlotItem(size=6)
        self.plotWidget_iq.addItem(self.iq_scatter)
        self.iq_colors = []
        
        top_graphs.addWidget(self.plotWidget_ted, 1)
        top_graphs.addWidget(self.plotWidget_iq, 1)
        left_layout.addLayout(top_graphs, 1)

        # 중간: Amplitude 그래프
        self.plotWidget_multi_data = PlotWidget()
        self.plotWidget_multi_data.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_multi_data.addLegend()
        self.plotWidget_multi_data.setTitle('Subcarrier Amplitude Data')
        self.plotWidget_multi_data.setLabel('left', 'Amplitude')
        self.plotWidget_multi_data.setLabel('bottom', 'Time (Cumulative Packet Count)')

        self.curve_list = []
        agc_curve = self.plotWidget_multi_data.plot(agc_gain_data, name='AGC Gain', pen=[255,255,0])
        fft_curve = self.plotWidget_multi_data.plot(fft_gain_data, name='FFT Gain', pen=[255,255,0])
        self.curve_list.append(agc_curve)
        self.curve_list.append(fft_curve)

        for i in range(CSI_DATA_COLUMNS):
            curve = self.plotWidget_multi_data.plot(self.csi_amplitude_array[:, i], name=str(i), pen=(255, 255, 255))
            self.curve_list.append(curve)
        
        left_layout.addWidget(self.plotWidget_multi_data, 1)

        # 하단: Phase 그래프
        self.plotWidget_phase_data = PlotWidget()
        self.plotWidget_phase_data.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        self.plotWidget_phase_data.addLegend()
        self.plotWidget_phase_data.setTitle('Subcarrier Phase Data')
        self.plotWidget_phase_data.setLabel('left', 'Phase (rad)')
        self.plotWidget_phase_data.setLabel('bottom', 'Time (Cumulative Packet Count)')

        self.curve_phase_list = []
        for i in range(CSI_DATA_COLUMNS):
            phase_curve = self.plotWidget_phase_data.plot(np.angle(self.csi_amplitude_array[:, i]), name=str(i), pen=(255, 255, 255))
            self.curve_phase_list.append(phase_curve)
        
        left_layout.addWidget(self.plotWidget_phase_data, 1)
        
        main_layout.addWidget(left_widget, 7)

        # === 오른쪽: Tello 패널 (비율 3) ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(8)
        
        # 연결 버튼 영역
        connect_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Tello 드론 연결")
        self.connect_btn.setMinimumHeight(50)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5a27;
                color: white;
                font-size: 20px;
                font-weight: bold;
                border: 2px solid #3d7a37;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #3d7a37; }
            QPushButton:pressed { background-color: #1d4a17; }
            QPushButton:disabled { background-color: #333333; color: #888888; }
        """)
        self.connect_btn.clicked.connect(self.connect_tello)
        
        self.connection_status_label = QLabel("연결되지 않음")
        self.connection_status_label.setStyleSheet("color: #888888; font-size: 18px; font-weight: bold;")
        self.connection_status_label.setAlignment(Qt.AlignCenter)
        
        connect_layout.addWidget(self.connect_btn, 1)
        connect_layout.addWidget(self.connection_status_label, 1)
        right_layout.addLayout(connect_layout)
        
        # Tello 카메라 영상
        # 비디오 레이블 제거 (성능 최적화)
        # self.video_label = QLabel("[ Tello 카메라 ]\n\n연결 버튼을 누러\n드론에 연결하세요")
        # self.video_label.setStyleSheet("color: #CCCCCC; background-color: #0d0d0d; font-size: 22px; font-weight: bold; border: 2px solid #333;")
        # self.video_label.setAlignment(Qt.AlignCenter)
        # self.video_label.setMinimumHeight(300)
        # right_layout.addWidget(self.video_label, 4)

        # Tello 상태 정보 표시
        self.status_group = QGroupBox("드론 상태")
        self.status_group.setStyleSheet("""
            QGroupBox {
                color: #CCCCCC;
                font-size: 22px;
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #0d0d0d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
            }
        """)
        
        status_layout = QVBoxLayout()
        status_layout.setSpacing(3)
        
        self.battery_label = QLabel("배터리: ---%")
        self.height_label = QLabel("높이: --- cm")
        self.temp_label = QLabel("온도: ---C")
        self.flight_time_label = QLabel("비행시간: --- s")
        self.barometer_label = QLabel("기압계: --- cm")
        self.tof_label = QLabel("ToF 거리: --- cm")
        self.attitude_label = QLabel("자세(P/R/Y): ---/---/---")
        self.speed_label = QLabel("속도(X/Y/Z): ---/---/---")
        self.accel_label = QLabel("가속도(X/Y/Z): ---/---/---")
        self.flight_status_label = QLabel("비행 상태: 착륙")
        
        # 어두운 배경 + 밝은 텍스트 (흰색 계열)
        default_label_style = "color: #FFFFFF; font-size: 18px; font-weight: bold; padding: 3px;"
        for label in [self.battery_label, self.height_label, self.temp_label, 
                      self.flight_time_label, self.barometer_label, self.tof_label,
                      self.attitude_label, self.speed_label, self.accel_label]:
            label.setStyleSheet(default_label_style)
            status_layout.addWidget(label)
        
        self.flight_status_label.setStyleSheet("color: #FFCC00; font-size: 20px; font-weight: bold; padding: 5px; background-color: #2a2a00; border-radius: 5px;")
        status_layout.addWidget(self.flight_status_label)
        
        self.status_group.setLayout(status_layout)
        right_layout.addWidget(self.status_group, 5)

        # 조종 안내
        self.control_group = QGroupBox("키보드 조종")
        self.control_group.setStyleSheet("""
            QGroupBox {
                color: #AAAAAA;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #444444;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #0d0d0d;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
            }
        """)
        
        control_layout = QVBoxLayout()
        controls_text = "[T] 이륙  |  [L] 착륙\n[W] 상승  |  [S] 하강  |  [A] 좌회전  |  [D] 우회전\n[↑] 전진  |  [↓] 후진  |  [←] 좌이동  |  [→] 우이동\n[Space] 비상정지"
        control_label = QLabel(controls_text)
        control_label.setStyleSheet("color: #CCCCCC; font-size: 16px; font-weight: bold; padding: 8px;")
        control_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(control_label)
        self.control_group.setLayout(control_layout)
        right_layout.addWidget(self.control_group, 2)
        
        main_layout.addWidget(right_widget, 3)

        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(150)
        self.deta_len = 0
        
        # 키 입력 활성화
        self.setFocusPolicy(Qt.StrongFocus)
        
        # 이미 연결된 Tello가 있으면 UI 업데이트
        if self.tello is not None:
            self.on_tello_connected(self.tello)
    
    def connect_tello(self):
        """Tello 드론 연결 시도 (버튼 클릭 시)"""
        self.connect_btn.setEnabled(False)
        self.connect_btn.setText("연결 중...")
        self.connection_status_label.setText("[ 연결 시도 중... ]")
        self.connection_status_label.setStyleSheet("color: #FFAA00; font-size: 18px; font-weight: bold;")
        # self.video_label.setText("Tello 드론에 연결 중...\n\n잠시만 기다려주세요")
        
        # 백그라운드 스레드에서 연결
        self.tello_connect_thread = TelloConnectThread()
        self.tello_connect_thread.connection_result.connect(self.on_connection_result)
        self.tello_connect_thread.start()
    
    def on_connection_result(self, tello, message):
        """Tello 연결 결과 처리"""
        if tello is not None:
            self.tello = tello
            self.on_tello_connected(tello)
            self.connection_status_label.setText(f"연결 성공!")
            self.connection_status_label.setStyleSheet("color: #88FF88; font-size: 18px; font-weight: bold;")
            self.connect_btn.setText("연결됨")
            self.connect_btn.setEnabled(False)
        else:
            self.connection_status_label.setText(f"연결 실패")
            self.connection_status_label.setStyleSheet("color: #FF8888; font-size: 18px; font-weight: bold;")
            self.connect_btn.setText("다시 연결")
            self.connect_btn.setEnabled(True)
            # self.video_label.setText(f"연결 실패\n\n{message}\n\n다시 연결을 시도해주세요")
    
    def on_tello_connected(self, tello):
        """Tello 연결 성공 시 상태 수집 스레드 시작"""
        self.tello = tello
        
        # 기존 스레드가 있으면 중지
        if self.tello_thread is not None:
            self.tello_thread.stop()
            self.tello_thread.wait()
        
        # 새 상태 수집 스레드 시작
        self.tello_thread = TelloThread(tello)
        self.tello_thread.state_updated.connect(self.update_tello_state)
        # self.tello_thread.frame_updated.connect(self.update_tello_frame)  # 비디오 비활성화
        self.tello_thread.start()

    def keyPressEvent(self, event):
        """키보드 입력 처리"""
        if self.tello is None or event.isAutoRepeat():
            return
            
        key = event.key()
        
        try:
            if key == Qt.Key_T:  # 이륙
                if not self.tello_flying:
                    self.tello.takeoff()
                    self.tello_flying = True
                    self.flight_status_label.setText("비행 상태: 비행중")
                    self.flight_status_label.setStyleSheet("color: #88FF88; font-weight: bold; font-size: 20px; padding: 5px; background-color: #1a3a1a;")
                    
            elif key == Qt.Key_L:  # 착륙
                if self.tello_flying:
                    self.tello.land()
                    self.tello_flying = False
                    self.flight_status_label.setText("비행 상태: 착륙")
                    self.flight_status_label.setStyleSheet("color: #FFCC00; font-size: 20px; font-weight: bold; padding: 5px; background-color: #2a2a00;")
                    
            elif key == Qt.Key_Space:  # 비상 정지
                self.tello.emergency()
                self.tello_flying = False
                self.flight_status_label.setText("비행 상태: 비상정지")
                self.flight_status_label.setStyleSheet("color: #FF6666; font-weight: bold; font-size: 20px; padding: 5px; background-color: #3a1a1a;")
            
            # 이동 키는 pressed_keys에 추가만 함
            elif key in [Qt.Key_W, Qt.Key_S, Qt.Key_A, Qt.Key_D,
                        Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
                self.pressed_keys.add(key)
                
        except Exception as e:
            print(f"Tello 명령 오류: {e}")
    
    def keyReleaseEvent(self, event):
        """키 릴리즈 시 키 상태에서 제거 및 드론 정지"""
        if self.tello is None or event.isAutoRepeat():
            return
            
        key = event.key()
        # 이동 관련 키가 떼어지면 pressed_keys에서 제거
        if key in self.pressed_keys:
            self.pressed_keys.discard(key)
            # 키를 뗀을 때 즉시 정지 명령 전송 (중요!)
            if not self.pressed_keys:  # 모든 키가 떼어진 경우
                try:
                    self.tello.send_rc_control(0, 0, 0, 0)
                except Exception as e:
                    print(f"Tello 정지 명령 오류: {e}")
    
    def send_control_commands(self):
        """타이머로 일정 간격마다 조종 명령 전송"""
        if self.tello is None or not self.pressed_keys:
            return
        
        speed = 50  # 이동 속도
        lr = fb = ud = yaw = 0  # left-right, forward-backward, up-down, yaw
        
        # 눌린 키들을 확인하여 속도 설정
        if Qt.Key_Left in self.pressed_keys:
            lr = -speed
        if Qt.Key_Right in self.pressed_keys:
            lr = speed
        if Qt.Key_Up in self.pressed_keys:
            fb = speed
        if Qt.Key_Down in self.pressed_keys:
            fb = -speed
        if Qt.Key_W in self.pressed_keys:
            ud = speed
        if Qt.Key_S in self.pressed_keys:
            ud = -speed
        if Qt.Key_A in self.pressed_keys:
            yaw = -speed
        if Qt.Key_D in self.pressed_keys:
            yaw = speed
        
        try:
            self.tello.send_rc_control(lr, fb, ud, yaw)
        except Exception as e:
            print(f"Tello RC 명령 오류: {e}")

    def update_tello_state(self, state):
        """Tello 상태 정보 업데이트 (가시성 향상)"""
        self.battery_label.setText(f"배터리: {state['battery']}%")
        self.height_label.setText(f"높이: {state['height']} cm")
        self.temp_label.setText(f"온도: {state['temperature']}C")
        self.flight_time_label.setText(f"비행시간: {state['flight_time']} s")
        self.barometer_label.setText(f"기압계: {state['barometer']:.1f} cm")
        self.tof_label.setText(f"ToF 거리: {state['tof']} cm")
        self.attitude_label.setText(f"자세(P/R/Y): {state['pitch']}/{state['roll']}/{state['yaw']}")
        self.speed_label.setText(f"속도(X/Y/Z): {state['speed_x']}/{state['speed_y']}/{state['speed_z']}")
        self.accel_label.setText(f"가속도(X/Y/Z): {state['acceleration_x']:.2f}/{state['acceleration_y']:.2f}/{state['acceleration_z']:.2f}")
        
        # 배터리 경고 (어두운 테마)
        if state['battery'] < 20:
            self.battery_label.setStyleSheet("color: #FF6666; font-size: 20px; font-weight: bold; padding: 3px; background-color: #330000;")
        elif state['battery'] < 50:
            self.battery_label.setStyleSheet("color: #FFCC66; font-size: 19px; font-weight: bold; padding: 3px;")
        else:
            self.battery_label.setStyleSheet("color: #FFFFFF; font-size: 18px; font-weight: bold; padding: 3px;")

    # 비디오 프레임 업데이트 함수 제거 (성능 최적화)
    # def update_tello_frame(self, frame):
    #     """Tello 카메라 영상 업데이트"""
    #     try:
    #         # BGR to RGB
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         h, w, ch = rgb_frame.shape
    #         bytes_per_line = ch * w
    #         
    #         # QImage로 변환
    #         q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    #         
    #         # 레이블 크기에 맞게 스케일 (FastTransformation으로 성능 향상)
    #         scaled_pixmap = QPixmap.fromImage(q_image).scaled(
    #             self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
    #         
    #         self.video_label.setPixmap(scaled_pixmap)
    #     except Exception as e:
    #         print(f"영상 표시 오류: {e}")

    def update_curve_colors(self, color_list):
        self.deta_len = len(color_list)
        self.iq_colors = color_list
        self.plotWidget_ted.setXRange(0, max(1, self.deta_len//2))
        # 범위 검사로 IndexError 방지
        max_curves = min(self.deta_len, len(self.curve_list), len(self.curve_phase_list))
        for i in range(max_curves):
            if i < len(color_list):
                self.curve_list[i].setPen(color_list[i])
                self.curve_phase_list[i].setPen(color_list[i])

    def update_data(self):
        # 최적화: 마지막 행만 계산 (필요한 경우에만)
        last_csi_amp = np.abs(csi_data_complex[-1, :])
        last_csi_phase = np.angle(csi_data_complex[-1, :])
        
        # IQ 플롯 업데이트
        i = np.real(csi_data_complex[-1, :])
        q = np.imag(csi_data_complex[-1, :])

        points = []
        for idx in range(min(self.deta_len, len(self.iq_colors))):
            color = self.iq_colors[idx]
            points.append({'pos': (i[idx], q[idx]), 'brush': pg.mkBrush(color)})

        self.iq_scatter.setData(points)

        # Phase 데이터 업데이트
        self.csi_row_data = last_csi_phase
        self.curve.setData(self.csi_row_data)

        # Gain 데이터 업데이트
        self.curve_list[CSI_DATA_COLUMNS].setData(agc_gain_data)
        self.curve_list[CSI_DATA_COLUMNS+1].setData(fft_gain_data)

        # Amplitude/Phase 업데이트 (전체 배열 계산 - 시각화에 필요)
        # 이 부분이 성능에 영향을 줄 수 있지만, 그래프 표시에 필요함
        for i in range(min(self.deta_len, CSI_DATA_COLUMNS)):
            self.curve_list[i].setData(np.abs(csi_data_complex[:, i]))
            self.curve_phase_list[i].setData(np.angle(csi_data_complex[:, i]))

def generate_subcarrier_colors(red_range, green_range, yellow_range, total_num,interval=1):
    colors = []
    for i in range(total_num):
        if red_range and red_range[0] <= i <= red_range[1]:
            intensity = int(255 * (i - red_range[0]) / (red_range[1] - red_range[0]))
            colors.append((intensity, 0, 0))
        elif green_range and green_range[0] <= i <= green_range[1]:
            intensity = int(255 * (i - green_range[0]) / (green_range[1] - green_range[0]))
            colors.append((0, intensity, 0))
        elif yellow_range and yellow_range[0] <= i <= yellow_range[1]:
            intensity = int(255 * (i - yellow_range[0]) / (yellow_range[1] - yellow_range[0]))
            colors.append((0, intensity, intensity))
        else:
            colors.append((200, 200, 200))

    return colors


def csi_data_read_parse(port: str, csv_writer, log_file_fd, callback=None):
    global fft_gains, agc_gains
    try:
        set = serial.Serial(port=port, baudrate=921600, bytesize=8, parity='N', stopbits=1)
    except Exception as e:
        print(f"Serial 포트 열기 실패: {e}")
        return
    
    count = 0
    if set.isOpen():
        print('Serial port open success')
    else:
        print('Serial port open failed')
        return
    
    last_save_time = 0
    while True:
        strings = str(set.readline())
        if not strings:
            break
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        index = strings.find('CSI_DATA')

        if index == -1:
            log_file_fd.write(strings + '\n')
            continue

        csv_reader = csv.reader(StringIO(strings))
        csi_data = next(csv_reader)
        csi_data_len = int (csi_data[-3])
        if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
            print('element number is not equal',len(csi_data),len(DATA_COLUMNS_NAMES) )
            log_file_fd.write('element number is not equal\n')
            log_file_fd.write(strings + '\n')
            continue

        try:
            csi_raw_data = json.loads(csi_data[-1])
        except json.JSONDecodeError:
            print('data is incomplete')
            log_file_fd.write('data is incomplete\n')
            log_file_fd.write(strings + '\n')
            continue
        if csi_data_len != len(csi_raw_data):
            print('csi_data_len is not equal',csi_data_len,len(csi_raw_data))
            log_file_fd.write('csi_data_len is not equal\n')
            log_file_fd.write(strings + '\n')
            continue

        fft_gain = int(csi_data[6])
        agc_gain = int(csi_data[7])

        # 메모리 누수 방지: 최대 1000개만 유지
        fft_gains.append(fft_gain)
        agc_gains.append(agc_gain)
        if len(fft_gains) > 1000:
            del fft_gains[:len(fft_gains)-1000]
            del agc_gains[:len(agc_gains)-1000]
        
        # 데이터 저장 속도 제한 (초당 20개: 0.05초 간격)
        current_time = time.time()
        if current_time - last_save_time >= 0.05:
            # 타임스탬프는 저장할 때만 생성 (성능 최적화)
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Tello 상태 데이터를 CSI 데이터와 함께 저장
            tello_row = [
                current_timestamp,
                tello_state_data['battery'],
                tello_state_data['height'],
                tello_state_data['temperature'],
                tello_state_data['flight_time'],
                tello_state_data['barometer'],
                tello_state_data['tof'],
                tello_state_data['pitch'],
                tello_state_data['roll'],
                tello_state_data['yaw'],
                tello_state_data['speed_x'],
                tello_state_data['speed_y'],
                tello_state_data['speed_z'],
                tello_state_data['acceleration_x'],
                tello_state_data['acceleration_y'],
                tello_state_data['acceleration_z'],
            ]
            combined_row = csi_data + tello_row
            csv_writer.writerow(combined_row)
            last_save_time = current_time

        # Rotate data to the left
        # csi_data_array[:-1] = csi_data_array[1:]
        # csi_data_phase[:-1] = csi_data_phase[1:]
        csi_data_complex[:-1] = csi_data_complex[1:]
        agc_gain_data[:-1] = agc_gain_data[1:]
        fft_gain_data[:-1] = fft_gain_data[1:]
        agc_gain_data[-1] = agc_gain
        fft_gain_data[-1] = fft_gain

        if count ==0:
            count = 1
            print('none',csi_data_len)
            if csi_data_len == 106:
                colors = generate_subcarrier_colors((0,25), (27,53), None, len(csi_raw_data))
            elif  csi_data_len == 114:
                colors = generate_subcarrier_colors((0,27), (29,56), None, len(csi_raw_data))
            elif  csi_data_len == 52:
                colors = generate_subcarrier_colors((0,12), (13,26), None, len(csi_raw_data))
            elif  csi_data_len == 234 :
                colors = generate_subcarrier_colors((0,28), (29,56), (60,116), len(csi_raw_data))
            elif  csi_data_len == 228 :
                colors = generate_subcarrier_colors((0,28), (29,57), (57,113), len(csi_raw_data))
            elif  csi_data_len == 490 :
                colors = generate_subcarrier_colors((0,61), (62,122), (123,245), len(csi_raw_data))
            elif  csi_data_len == 128 :
                colors = generate_subcarrier_colors((0,31), (32,63), None, len(csi_raw_data))
            elif  csi_data_len == 256 :
                colors = generate_subcarrier_colors((0,32), (32,63), (64,128), len(csi_raw_data))
            elif  csi_data_len == 512 :
                colors = generate_subcarrier_colors((0,63), (64,127), (128,256), len(csi_raw_data))
            elif  csi_data_len == 384 :
                colors = generate_subcarrier_colors((0,63), (64,127), (128,192), len(csi_raw_data))
            elif csi_data_len > 0 and csi_data_len <= 612:
                raw_len = len(csi_raw_data)
                colors = generate_subcarrier_colors((0,raw_len//2), (raw_len//2+1,raw_len-1), None, raw_len)
            if callback is not None:
                callback(colors)

        for i in range(csi_data_len // 2):
            csi_data_complex[-1][i] = complex(csi_raw_data[i * 2 + 1],
                                            csi_raw_data[i * 2])
    set.close()
    return


class SubThread (QThread):
    data_ready = pyqtSignal(object)
    def __init__(self, serial_port, save_file_name, log_file_name):
        super().__init__()
        self.serial_port = serial_port

        # 버퍼링을 크게 설정하여 디스크 IO 감소 (8KB)
        self.save_file_fd = open(save_file_name, 'w', buffering=8192)
        self.log_file_fd = open(log_file_name, 'w', buffering=8192)
        self.csv_writer = csv.writer(self.save_file_fd)
        # CSI + Tello 컬럼명 모두 저장
        self.csv_writer.writerow(COMBINED_COLUMNS_NAMES)

    def run(self):
        csi_data_read_parse(self.serial_port, self.csv_writer, self.log_file_fd,callback=self.data_ready.emit)

    def __del__(self):
        self.wait()
        # 파일 명시적 닫기
        if hasattr(self, 'save_file_fd'):
            self.save_file_fd.close()
        if hasattr(self, 'log_file_fd'):
            self.log_file_fd.close()


if __name__ == '__main__':
    if sys.version_info < (3, 6):
        print(' Python version should >= 3.6')
        exit()

    parser = argparse.ArgumentParser(
        description='Read CSI data from serial port and display it graphically with Tello drone control')
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help='Serial port number of csv_recv device')
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_tello_data.csv',
                        help='Save the data printed by the serial port to a file')
    parser.add_argument('-l', '--log', dest='log_file', action='store', default='./csi_data_log.txt',
                        help='Save other serial data the bad CSI data to a log file')
    parser.add_argument('--no-tello', dest='no_tello', action='store_true', default=False,
                        help='Run without Tello drone connection')

    args = parser.parse_args()
    serial_port = args.port
    file_name = args.store_file
    log_file_name = args.log_file

    app = QApplication(sys.argv)

    # CSI 데이터 수집 스레드
    subthread = SubThread(serial_port, file_name, log_file_name)

    # GUI 윈도우 생성 (Tello는 나중에 연결 버튼으로 연결)
    window = csi_data_graphical_window(tello=None)
    subthread.data_ready.connect(window.update_curve_colors)
    subthread.start()
    
    print("GUI가 시작되었습니다. Tello 연결 버튼을 눌러 드론에 연결하세요.")
    
    window.show()
    
    # 종료 시 정리
    def cleanup():
        if window.tello_thread is not None:
            window.tello_thread.stop()
            window.tello_thread.wait()
        if window.tello is not None:
            try:
                window.tello.streamoff()
                window.tello.end()
            except:
                pass
    
    app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec())