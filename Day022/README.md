# Day022 — YOLO 첫 학습 및 측정 ROI 자동 검출

## 목표

Day021에서 생성한 measurement_roi 라벨 데이터를 이용하여
YOLO 모델을 처음 학습하고 측정 ROI 자동 검출을 테스트한다.

## 학습 데이터

Train:
- sample_03
- sample_04
- sample_06
- sample_N01
- sample_N02

Validation:
- sample_07

## 학습

YOLO26n 사전학습 모델 사용

1차 학습:
- 30 epochs
- ROI 검출 실패

2차 학습:
- 100 epochs

Validation 결과:
- Recall: 1.000
- mAP50: 0.995
- mAP50-95: 0.801

## 실제 Predict 테스트

일반 confidence 0.25:
- ROI 검출 실패

진단용 confidence 0.001:
- ROI 후보 검출 성공
- confidence 약 0.008 수준

## 분석

YOLO 학습 및 추론 파이프라인은 정상적으로 구현되었다.

그러나 현재 데이터가 Train 5장 / Validation 1장으로 매우 적어
실제 추론 신뢰도가 낮게 나타났다.

따라서 confidence 값을 강제로 낮추는 것보다
추가 학습 데이터를 확보하여 모델의 ROI 검출 신뢰도를 높이는 것이 필요하다.

## 다음 단계

측정 ROI 이미지 데이터 확대
→ YOLO 재학습
→ confidence 개선
→ YOLO ROI 자동 Crop
→ OpenCV Sub-pixel 외경 측정 연결