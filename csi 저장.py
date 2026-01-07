import serial
import csv
import time
import re

# --- 설정 구간 ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 921600
TARGET_MAC = "FA:29:AA:6F:BF:24"  # 시각화 중인 핫스팟 주소
FILE_NAME = "hotspot_csi_log.csv"

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        ser.reset_input_buffer()
        print(f"[{TARGET_MAC}] 데이터 기록 시작... (저장처: {FILE_NAME})")
        
        with open(FILE_NAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 헤더 작성 (타임스탬프, MAC 주소, RSSI, CSI 데이터 전체)
            writer.writerow(["Timestamp", "MAC", "RSSI", "CSI_Raw_Data"])
            
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # 지정한 MAC 주소 패킷만 필터링
                    if "CSI_DATA" in line and TARGET_MAC.upper() in line.upper():
                        # RSSI 및 CSI 데이터 추출을 위한 파싱
                        parts = line.split(',')
                        rssi = parts[3] if len(parts) > 3 else "N/A"
                        csi_match = re.search(r'\[(.*?)\]', line)
                        
                        if csi_match:
                            csi_raw = csi_match.group(1)
                            # 현재 시간과 함께 저장
                            writer.writerow([time.time(), TARGET_MAC, rssi, csi_raw])
                            print(f"Saved packet at {time.strftime('%H:%M:%S')} (RSSI: {rssi})")

    except KeyboardInterrupt:
        print(f"\n기록이 중단되었습니다. 파일 '{FILE_NAME}'을 확인하세요.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()