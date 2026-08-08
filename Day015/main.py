import cv2
import numpy as np
from pathlib import Path


# 1. Day013 ROI 이미지 읽기
image_path = Path("Day013/shaft_roi.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("ROI 이미지를 찾을 수 없습니다.")


# 2. GRAY 변환
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# 3. Blur
blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# 4. Edge 검출
edge = cv2.Canny(
    blur,
    50,
    150
)


# 5. 이미지 크기
height, width = edge.shape

print("ROI width :", width)
print("ROI height :", height)


# 6. 측정할 중앙 영역 지정
x_start = int(width * 0.25)
x_end = int(width * 0.75)

print("측정 시작 x :", x_start)
print("측정 끝 x :", x_end)


# 7. 여러 x 위치 생성
sample_count = 7

x_positions = np.linspace(
    x_start,
    x_end,
    sample_count,
    dtype=int
)

print("측정 x 위치 :", x_positions)


# 8. 측정값 저장 리스트
diameters = []

result = image.copy()


# 9. 여러 위치에서 외경 측정
for x in x_positions:

    # x 위치 주변 5pixel 폭 사용
    x1 = max(0, x - 2)
    x2 = min(width, x + 3)

    strip = edge[:, x1:x2]

    # 각 y 위치의 Edge 개수
    row_scores = np.sum(
        strip > 0,
        axis=1
    )

    center_y = height // 2

    # 위쪽 Edge
    top_scores = row_scores[:center_y]

    y_top = np.argmax(
        top_scores
    )

    # 아래쪽 Edge
    bottom_scores = row_scores[center_y:]

    y_bottom = center_y + np.argmax(
        bottom_scores
    )

    # 외경 계산
    diameter = y_bottom - y_top

    diameters.append(diameter)

    print(
        "x :", x,
        "top :", y_top,
        "bottom :", y_bottom,
        "diameter :", diameter
    )


    # 측정선 표시
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


# 10. 평균 / 중앙값
mean_diameter = np.mean(
    diameters
)

median_diameter = np.median(
    diameters
)


print()
print("측정값 :", diameters)
print("평균 외경(pixel) :", mean_diameter)
print("중앙값 외경(pixel) :", median_diameter)


# 11. 결과 이미지 저장
output_dir = Path("Day015")
output_dir.mkdir(exist_ok=True)

edge_path = output_dir / "edge.jpg"

result_path = output_dir / "multi_measure_result.jpg"

cv2.imwrite(
    str(edge_path),
    edge
)

cv2.imwrite(
    str(result_path),
    result
)

print()
print("저장 완료 :", result_path)