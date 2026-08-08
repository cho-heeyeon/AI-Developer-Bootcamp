# Day020 — 다점 Calibration 보정식 만들기

## 목표

단일 기준점 Calibration에서 발전하여
여러 개의 Pixel 측정값과 실제 계측값을 이용해
다점 Linear Calibration 보정식을 만든다.

보정식:

mm = a × pixel + b


## 구현 내용

- 여러 Calibration 데이터 입력
- `np.polyfit()`을 이용한 1차 보정식 계산
- Calibration 데이터의 평균 절대오차 계산
- 최대 절대오차 계산
- 새로운 Sub-pixel 측정값을 mm로 변환
- 실제 계측값과 비교하여 오차 계산
- ±10 μm PASS / FAIL 판정


## 실행 결과

보정식:

mm = 0.0433810902 × pixel + 0.1001849107

Calibration 평균 절대오차:

약 0.152 μm

Calibration 최대 절대오차:

약 0.303 μm


검증 예제 결과:

영상 측정값:
20.028156 mm

실제 계측값:
20.022 mm

절대오차:
약 6.156 μm

±10 μm 기준:
PASS


## 주의사항

이번 Day020에서 사용한 Calibration 값은
다점 Calibration 프로그램의 동작을 확인하기 위한 예제 데이터이다.

따라서 6.156 μm PASS 결과를
실제 측정 시스템의 최종 정확도로 판단하면 안 된다.


## 실제 데이터 적용 과정에서 확인된 문제

실제 샤프트 이미지에 Sub-pixel 측정을 적용한 결과
샘플별 Pixel 값의 편차가 크게 나타났다.

예:

03  : 약 460.245 pixel
04  : 약 383.830 pixel
06  : 약 460.112 pixel
N01 : 약 404.742 pixel
N02 : 약 401.892 pixel
07  : 약 274.470 pixel

실제 외경 차이에 비해 Pixel 측정값 차이가 지나치게 커서
Calibration 전에 Edge 검출 안정화가 필요하다고 판단하였다.


## 분석 결과

Day020-1-1 디버깅 결과,
하단 Edge는 비교적 안정적으로 검출되었으나
상단 Edge가 금속 표면 내부의 반사 Edge를 선택하는 문제가 확인되었다.

따라서 현재 단계에서는
잘못된 Pixel 값을 이용하여 Calibration을 진행하면 안 된다.


## 결론

다점 Calibration 프로그램 구조는 정상적으로 구현하였다.

그러나 실제 ±10 μm 측정을 위해서는
Calibration 이전에 측정 ROI와 Edge 검출의 반복성을 확보해야 한다.


## 다음 단계

YOLO를 이용하여 측정 ROI를 자동으로 안정적으로 검출하고,

YOLO ROI
→ OpenCV Edge
→ Sub-pixel
→ 다점 Calibration
→ 실제 외경(mm)

구조로 발전시킨다.