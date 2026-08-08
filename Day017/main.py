import cv2
import numpy as np
from pathlib import Path


# ---------------------------------
# 1. Day016에서 구한 캘리브레이션 값
# ---------------------------------

MM_PER_PIXEL = 0.0436166


# ---------------------------------
# 2. 측정 이미지 불러오기
# ---------------------------------

image_path = Path("Day013/shaft_roi.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("측정 이미지를 찾을 수 없습니다.")


# ---------------------------------
# 3. GRAY 변환
# ---------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ---------------------------------
# 4. Blur
# ---------------------------------

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# ---------------------------------
# 5. Edge 검출
# ---------------------------------

edge = cv2.Canny(
    blur,
    50,
    150
)


# ---------------------------------
# 6. 이미지 크기
# ---------------------------------

height, width = edge.shape

center_y = height // 2


# ---------------------------------
# 7. 중앙 측정 영역
# ---------------------------------

x_start = int(width * 0.25)
x_end = int(width * 0.75)

sample_count = 7

x_positions = np.linspace(
    x_start,
    x_end,
    sample_count,
    dtype=int
)


# ---------------------------------
# 8. 여러 위치에서 외경 측정
# ---------------------------------

diameters = []

result = image.copy()


for x in x_positions:

    x1 = max(0, x - 2)
    x2 = min(width, x + 3)

    strip = edge[:, x1:x2]

    row_scores = np.sum(
        strip > 0,
        axis=1
    )


    # 위쪽 Edge
    top_scores = row_scores[:center_y]

    y_top = np.argmax(top_scores)


    # 아래쪽 Edge
    y_bottom = None

    for y in range(center_y, height):

        if row_scores[y] >= 2:
            y_bottom = y


    if y_bottom is None:
        continue


    diameter_pixel = y_bottom - y_top

    diameters.append(
        diameter_pixel
    )


    # 결과 표시
    cv2.circle(
        result,
        (x, y_top),
        5,
        (0, 0, 255),
        -1
    )

    cv2.circle(
        result,
        (x, y_bottom),
        5,
        (255, 0, 0),
        -1
    )

    cv2.line(
        result,
        (x, y_top),
        (x, y_bottom),
        (0, 255, 0),
        2
    )


# ---------------------------------
# 9. 대표 pixel 외경
# ---------------------------------

median_pixel = np.median(
    diameters
)

print("측정값(pixel) :", diameters)
print("대표 외경(pixel) :", median_pixel)


# ---------------------------------
# 10. Pixel → mm 변환
# ---------------------------------

measured_mm = (
    median_pixel
    * MM_PER_PIXEL
)

print("영상 측정 외경(mm) :", measured_mm)


# ---------------------------------
# 11. 실제 계측기값 입력
# ---------------------------------

actual_mm = float(
    input("실제 계측기 외경(mm)을 입력하세요 : ")
)


# ---------------------------------
# 12. 오차 계산
# ---------------------------------

error_mm = (
    measured_mm
    - actual_mm
)

error_um = (
    error_mm
    * 1000
)

abs_error_um = abs(
    error_um
)


print()
print("영상 측정값(mm) :", measured_mm)
print("실제 계측값(mm) :", actual_mm)

print("오차(mm) :", error_mm)
print("오차(μm) :", error_um)
print("절대오차(μm) :", abs_error_um)


# ---------------------------------
# 13. 10 μm 기준 판정
# ---------------------------------

if abs_error_um <= 10:

    judgment = "PASS"

else:

    judgment = "FAIL"


print("10 μm 기준 :", judgment)


# ---------------------------------
# 14. 이미지에 결과 표시
# ---------------------------------

text1 = f"Vision: {measured_mm:.3f} mm"

text2 = f"Actual: {actual_mm:.3f} mm"

text3 = f"Error: {error_um:.1f} um"

text4 = f"Result: {judgment}"


cv2.putText(
    result,
    text1,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    result,
    text2,
    (10, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    result,
    text3,
    (10, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.putText(
    result,
    text4,
    (10, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)


# ---------------------------------
# 15. 저장
# ---------------------------------

output_dir = Path("Day017")

output_dir.mkdir(exist_ok=True)

cv2.imwrite(
    str(output_dir / "edge.jpg"),
    edge
)

cv2.imwrite(
    str(output_dir / "validation_result.jpg"),
    result
)

print()
print("저장 완료 : Day017/validation_result.jpg")