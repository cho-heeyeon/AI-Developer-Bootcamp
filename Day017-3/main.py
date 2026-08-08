import cv2
import numpy as np
from pathlib import Path


# ---------------------------------
# 1. 검증 ROI 이미지
# ---------------------------------

image_path = Path("Day017-2/test_roi.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("검증 ROI 이미지를 찾을 수 없습니다.")


# ---------------------------------
# 2. GRAY / Blur / Edge
# ---------------------------------

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

edge = cv2.Canny(
    blur,
    50,
    150
)


# ---------------------------------
# 3. 이미지 크기
# ---------------------------------

height, width = edge.shape

center_y = height // 2

print("ROI width :", width)
print("ROI height :", height)
print("center_y :", center_y)


# ---------------------------------
# 4. 중앙 측정 영역
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
# 5. 결과 저장용 리스트
# ---------------------------------

top_values = []
bottom_values = []
diameters = []

result = image.copy()


# ---------------------------------
# 6. 각 x 위치 분석
# ---------------------------------

print()
print("x | top | bottom | diameter")
print("--------------------------------")


for x in x_positions:

    x1 = max(0, x - 2)
    x2 = min(width, x + 3)

    strip = edge[:, x1:x2]

    row_scores = np.sum(
        strip > 0,
        axis=1
    )


    # 상단 Edge
    top_scores = row_scores[:center_y]

    y_top = np.argmax(
        top_scores
    )


    # 하단 Edge
    y_bottom = None

    for y in range(center_y, height):

        if row_scores[y] >= 2:
            y_bottom = y


    if y_bottom is None:
        continue


    diameter = y_bottom - y_top


    top_values.append(y_top)
    bottom_values.append(y_bottom)
    diameters.append(diameter)


    print(
        x,
        "|",
        y_top,
        "|",
        y_bottom,
        "|",
        diameter
    )


    # ---------------------------------
    # 측정점 표시
    # ---------------------------------

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
# 7. 통계 계산
# ---------------------------------

top_mean = np.mean(top_values)
bottom_mean = np.mean(bottom_values)

diameter_mean = np.mean(diameters)
diameter_median = np.median(diameters)

top_std = np.std(top_values)
bottom_std = np.std(bottom_values)
diameter_std = np.std(diameters)


print()
print("========== 분석 결과 ==========")

print("상단 Edge 평균 :", top_mean)
print("상단 Edge 표준편차 :", top_std)

print("하단 Edge 평균 :", bottom_mean)
print("하단 Edge 표준편차 :", bottom_std)

print("외경 평균(pixel) :", diameter_mean)
print("외경 중앙값(pixel) :", diameter_median)
print("외경 표준편차(pixel) :", diameter_std)


# ---------------------------------
# 8. Day016 기준값과 비교
# ---------------------------------

BASE_PIXEL = 459

pixel_difference = (
    diameter_median
    - BASE_PIXEL
)

print()
print("Day016 기준(pixel) :", BASE_PIXEL)
print("Day017-3 대표값(pixel) :", diameter_median)
print("차이(pixel) :", pixel_difference)


# ---------------------------------
# 9. μm 환산
# ---------------------------------

MM_PER_PIXEL = 0.0436166

difference_mm = (
    pixel_difference
    * MM_PER_PIXEL
)

difference_um = (
    difference_mm
    * 1000
)

print("차이(mm) :", difference_mm)
print("차이(μm) :", difference_um)


# ---------------------------------
# 10. 저장
# ---------------------------------

output_dir = Path("Day017-3")

output_dir.mkdir(exist_ok=True)


cv2.imwrite(
    str(output_dir / "edge.jpg"),
    edge
)

cv2.imwrite(
    str(output_dir / "edge_analysis_result.jpg"),
    result
)

print()
print("저장 완료 : Day017-3/edge_analysis_result.jpg")