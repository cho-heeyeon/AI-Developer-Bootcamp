import cv2
import numpy as np
from pathlib import Path


# 1. ROI 이미지 읽기
image_path = Path("Day013/shaft_roi.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("ROI 이미지를 찾을 수 없습니다.")


# 2. GRAY 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 3. Blur
blur = cv2.GaussianBlur(gray, (5, 5), 0)


# 4. Edge 검출
edge = cv2.Canny(blur, 50, 150)


# 5. 이미지 크기
height, width = edge.shape

center_y = height // 2

print("ROI width :", width)
print("ROI height :", height)
print("center_y :", center_y)


# 6. 중앙 측정 영역
x_start = int(width * 0.25)
x_end = int(width * 0.75)

sample_count = 7

x_positions = np.linspace(
    x_start,
    x_end,
    sample_count,
    dtype=int
)


diameters = []

result = image.copy()


# 7. 여러 x 위치에서 측정
for x in x_positions:

    x1 = max(0, x - 2)
    x2 = min(width, x + 3)

    strip = edge[:, x1:x2]

    row_scores = np.sum(
        strip > 0,
        axis=1
    )


    # -------------------------
    # 상단 Edge
    # -------------------------

    top_scores = row_scores[:center_y]

    y_top = np.argmax(top_scores)


    # -------------------------
    # 하단 Edge 개선
    # -------------------------

    y_bottom = None

    # 중심에서 아래쪽으로 탐색
    for y in range(center_y, height):

        # 이 y 위치에 Edge가 충분히 있으면
        if row_scores[y] >= 2:

            y_bottom = y


    # Edge를 못 찾은 경우
    if y_bottom is None:

        print("하단 Edge 검출 실패 :", x)
        continue


    # 외경 계산
    diameter = y_bottom - y_top

    diameters.append(diameter)


    print(
        "x :", x,
        "top :", y_top,
        "bottom :", y_bottom,
        "diameter :", diameter
    )


    # 상단 점
    cv2.circle(
        result,
        (x, y_top),
        5,
        (0, 0, 255),
        -1
    )


    # 하단 점
    cv2.circle(
        result,
        (x, y_bottom),
        5,
        (255, 0, 0),
        -1
    )


    # 외경 측정선
    cv2.line(
        result,
        (x, y_top),
        (x, y_bottom),
        (0, 255, 0),
        2
    )


# 8. 평균 / 중앙값
if len(diameters) > 0:

    mean_diameter = np.mean(diameters)

    median_diameter = np.median(diameters)

    print()
    print("측정값 :", diameters)
    print("평균 외경(pixel) :", mean_diameter)
    print("중앙값 외경(pixel) :", median_diameter)


# 9. 저장
output_dir = Path("Day015_2")

output_dir.mkdir(exist_ok=True)

cv2.imwrite(
    str(output_dir / "edge.jpg"),
    edge
)

cv2.imwrite(
    str(output_dir / "bottom_edge_result.jpg"),
    result
)

print()
print("저장 완료 :", output_dir / "bottom_edge_result.jpg")