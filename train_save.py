import pandas as pd
import numpy as np
import ast
import joblib
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# [핵심] 특징 추출 함수 (학습/실시간 공용)
# ==========================================
def extract_features(csi_list):
    """
    128개의 CSI 데이터에서 9개의 통계적 특징을 추출합니다.
    이 함수는 학습할 때와 실시간 예측할 때 똑같이 사용해야 합니다.
    """
    if len(csi_list) == 0:
        return [0] * 9 # 예외 처리

    data = np.array(csi_list)
    
    # 1. 기본 통계
    mean_val = np.mean(data)       # 평균 신호 세기
    std_val = np.std(data)         # 표준편차 (흔들림 정도 - 중요!)
    max_val = np.max(data)         # 최대값
    min_val = np.min(data)         # 최소값
    range_val = max_val - min_val  # 변동 폭
    
    # 2. 분포 형태 (벽 반사파가 생기면 분포가 찌그러짐)
    skew_val = stats.skew(data)    # 왜도 (치우침)
    kurt_val = stats.kurtosis(data)# 첨도 (뾰족함)
    
    # 3. 추가 통계
    # 사분위수 범위 (IQR) - 이상치에 강한 변동성 지표
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    
    # 에너지 (신호 총량)
    energy = np.sum(data ** 2) / len(data)

    return [mean_val, std_val, max_val, min_val, range_val, skew_val, kurt_val, iqr, energy]

# ==========================================
# 메인 학습 로직
# ==========================================
if __name__ == "__main__":
    print("1. 데이터 로딩 중...")
    try:
        df_hover = pd.read_csv('공중 호버.csv')
        df_wall = pd.read_csv('벽 가까이.csv')
    except FileNotFoundError:
        print("CSV 파일이 없습니다. 경로를 확인해주세요.")
        exit()

    # 라벨링
    df_hover['label'] = 0
    df_wall['label'] = 1

    # 파싱 함수
    def parse_csi(csi_str):
        try: return ast.literal_eval(csi_str)
        except: return []

    print("2. 데이터 파싱 및 특징 추출 중 (시간이 조금 걸릴 수 있습니다)...")
    
    # (1) 파싱
    df_hover['csi_raw'] = df_hover['data'].apply(parse_csi)
    df_wall['csi_raw'] = df_wall['data'].apply(parse_csi)

    # (2) 길이 필터링 (노이즈 제거)
    df_hover = df_hover[df_hover['csi_raw'].apply(len) == 128]
    df_wall = df_wall[df_wall['csi_raw'].apply(len) == 128]

    # (3) 특징 추출 적용
    # 데이터프레임의 각 행에 대해 extract_features 함수 실행
    X_hover = [extract_features(row) for row in df_hover['csi_raw']]
    X_wall = [extract_features(row) for row in df_wall['csi_raw']]
    
    # (4) 시간순 분할 (Time Series Split) - 섞지 않음!
    def split_data(data_list, label, ratio=0.8):
        split_idx = int(len(data_list) * ratio)
        return data_list[:split_idx], data_list[split_idx:], [label]*split_idx, [label]*(len(data_list)-split_idx)

    X_train_h, X_test_h, y_train_h, y_test_h = split_data(X_hover, 0)
    X_train_w, X_test_w, y_train_w, y_test_w = split_data(X_wall, 1)

    # 합치기
    X_train = np.array(X_train_h + X_train_w)
    y_train = np.array(y_train_h + y_train_w)
    X_test = np.array(X_test_h + X_test_w)
    y_test = np.array(y_test_h + y_test_w)

    print(f"학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
    print(f"특징 개수: {X_train.shape[1]}개 (Mean, Std, Skew, etc.)")

    # 3. 모델 학습
    print("3. 모델 학습 중...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,       # 과적합 방지
        min_samples_leaf=4, # 안정성 강화
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 4. 평가
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n[결과] 실전 예상 정확도: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Hover', 'Wall']))

    # 5. 저장
    joblib.dump(clf, 'csi_model_advanced.pkl')
    print("모델 저장 완료: csi_model_advanced.pkl")