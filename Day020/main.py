import numpy as np
from pathlib import Path


# -----------------------------------
# 1. Calibration 데이터
# -----------------------------------
# 아래 숫자는 예시입니다.
# 실제 측정 데이터가 준비되면 반드시 교체합니다.

pixel_values = np.array([
    458.72,
    459.18,
    459.65,
    460.10
])

actual_mm_values = np.array([
    20.000,
    20.020,
    20.040,
    20.060
])


# -----------------------------------
# 2. 다점 Linear Calibration
# -----------------------------------

a, b = np.polyfit(
    pixel_values,
    actual_mm_values,
    1
)


print("Calibration 결과")
print("-----------------------------")

print("기울기 a :", a)
print("절편 b :", b)

print()

print(
    f"보정식 : mm = {a:.10f} × pixel + {b:.10f}"
)


# -----------------------------------
# 3. Calibration 데이터 자체 확인
# -----------------------------------

predicted_mm = (
    a * pixel_values
    + b
)


errors_mm = (
    predicted_mm
    - actual_mm_values
)


errors_um = (
    errors_mm * 1000
)


print()
print("Calibration 데이터 확인")
print("-----------------------------")


for i in range(len(pixel_values)):

    print(
        f"{i+1}번 | "
        f"pixel={pixel_values[i]:.3f} | "
        f"실제={actual_mm_values[i]:.3f} mm | "
        f"계산={predicted_mm[i]:.3f} mm | "
        f"오차={errors_um[i]:.2f} um"
    )


# -----------------------------------
# 4. 전체 Calibration 오차
# -----------------------------------

mae_um = np.mean(
    np.abs(errors_um)
)

max_error_um = np.max(
    np.abs(errors_um)
)


print()
print("평균 절대오차 :", mae_um, "um")
print("최대 절대오차 :", max_error_um, "um")


# -----------------------------------
# 5. 새로운 Sub-pixel 측정값 입력
# -----------------------------------

test_pixel = float(
    input(
        "\n새로운 영상 Sub-pixel 외경값을 입력하세요 : "
    )
)


# -----------------------------------
# 6. Pixel → mm 변환
# -----------------------------------

vision_mm = (
    a * test_pixel
    + b
)


print()
print("영상 측정값 :", vision_mm, "mm")


# -----------------------------------
# 7. 실제 계측값 입력
# -----------------------------------

actual_test_mm = float(
    input(
        "새 데이터의 실제 계측값(mm)을 입력하세요 : "
    )
)


# -----------------------------------
# 8. 검증 오차
# -----------------------------------

error_mm = (
    vision_mm
    - actual_test_mm
)

error_um = (
    error_mm * 1000
)

abs_error_um = abs(
    error_um
)


print()
print("========== 검증 결과 ==========")

print(
    "영상 측정값(mm) :",
    vision_mm
)

print(
    "실제 계측값(mm) :",
    actual_test_mm
)

print(
    "오차(mm) :",
    error_mm
)

print(
    "오차(um) :",
    error_um
)

print(
    "절대오차(um) :",
    abs_error_um
)


# -----------------------------------
# 9. ±10 μm 판정
# -----------------------------------

if abs_error_um <= 10:

    judgment = "PASS"

else:

    judgment = "FAIL"


print(
    "±10 um 기준 :",
    judgment
)