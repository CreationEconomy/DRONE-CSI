### DRONE-CSI (Tello WiFi CSI 확인)

목표는 **DJI Tello(AP)의 WiFi 신호를 ESP32가 받아 CSI(Channel State Information)를 출력**하고, PC에서 `visualize_csi.py`로 확인하는 것입니다.  
Tello는 부팅 때마다 채널이 바뀔 수 있어서, 펌웨어 쪽은 **TELLO- 자동 탐색(Fast Active Scan)**이 핵심입니다.

### 추천 워크플로우(가장 성공 확률 높음)
- **1순위: Passive(스니핑)로 “CSI_DATA가 뜨는지” 먼저 확인**
  - 이유: 드론이 다운링크 트래픽을 거의 안 내면 Station 연결만으로는 CSI가 잘 안 뜰 수 있음
  - 경로: `../esp32-csi-tool/passive`
  - 동작: `TELLO-`를 빠르게 스캔(채널당 30~60ms) → 최적 채널로 고정 → 프로미스큐어스(CSI) 수집

- **2순위: Station으로 Tello에 붙어서 CSI 확인**
  - 경로: `../esp32-csi-tool/active_sta`
  - 동작: `TELLO-` 빠른 스캔 → 최적 AP 선택(강한 RSSI) → 채널/SSID 반영 후 연결

### ESP32 펌웨어(ESP-IDF) 빌드/플래시
아래 둘 중 하나를 선택해서 진행하세요.

- **Passive(추천)**:

```bash
cd /home/clip968/Desktop/programming/esp32-csi-tool/passive
idf.py menuconfig
idf.py -p /dev/ttyUSB0 build flash monitor
```

- **Station(연결 모드)**:

```bash
cd /home/clip968/Desktop/programming/esp32-csi-tool/active_sta
idf.py menuconfig
idf.py -p /dev/ttyUSB0 build flash monitor
```

#### menuconfig에서 꼭 확인할 것
- **Baudrate**: `921600` (모니터/콘솔 UART 둘 다)
- **WiFi CSI 활성화**: `Component config > Wi-Fi > WiFi CSI(Channel State Information)` 체크
- **(Station만)**: 드론에 비밀번호가 설정돼 있으면 `ESP_WIFI_PASSWORD`를 맞게 설정

### 정상 동작 확인(ESP32 로그)
모니터에서 아래 두 가지가 보이면 목표의 80%는 달성입니다.
- **TELLO 스캔 결과 로그**: `Best TELLO AP: SSID:... CH:... RSSI:... BSSID:...`
- **CSI 출력**: `CSI_DATA,...,[ ... ]` 라인이 계속 출력

### PC에서 CSI 시각화/저장(Python)
기본 포트는 `/dev/ttyUSB0`, 보드는 `921600` baud를 사용합니다.

- **실시간 시각화**:

```bash
python /home/clip968/Desktop/programming/DRONE-CSI/visualize_csi.py
```

- **CSV 저장**:

```bash
python /home/clip968/Desktop/programming/DRONE-CSI/csi\ 저장.py
```

#### 중요: MAC 필터(TARGET_MAC)
`visualize_csi.py`, `csi 저장.py`는 이제 **기본값이 `TARGET_MAC = None`(필터 해제)**입니다.  
특정 드론만 보고 싶으면, ESP32 로그에 찍힌 `BSSID`(또는 `nmcli dev wifi`의 BSSID)를 `TARGET_MAC`에 넣으세요.

### 트러블슈팅(자주 막히는 포인트)
- **CSI_DATA가 안 뜸**
  - **Passive로 먼저 확인**(드론이 트래픽이 적으면 Station에서 CSI가 거의 안 나올 수 있음)
  - **`TARGET_MAC`이 None인지 확인**(필터 걸려 있으면 “데이터 0”처럼 보임)
  - **거리/안테나**: 드론 가까이에서 테스트(초기엔 0.5~2m 권장)
- **부팅/재연결이 느림**
  - 현재 코드는 `TELLO-` **Fast Active Scan(30~60ms/ch)**로 채널을 빠르게 잡도록 되어 있어야 함