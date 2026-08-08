import cv2
import numpy as np
from pathlib import Path


# -----------------------------
# 1. 비교할 이미지
# -----------------------------

image_016_path = Path("Day013/shaft_roi_box.jpg")
image_017_path = Path("Day017/test_roi.jpg")


image_016 = cv2.imread(str(image_016_path))
image_017 = cv2.imread(str(image_017_path))

if image_016 is None:
    raise FileNotFoundError("Day016 기준 ROI 이미지를 찾을 수 없습니다.")

if image_017 is None:
    raise FileNotFoundError("Day017 검증 ROI 이미지를 찾을 수 없습니다.")


# -----------------------------
# 2. 공통 측정 함수
# -----------------------------

def measure_diameter(image, name):

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

    height, width = edge.shape

    center_y = height // 2

    x_start = int(width * 0.25)
    x_end = int(width * 0.75)

    x_positions = np.linspace(
        x_start,
        x_end,
        7,
        dtype=int
    )

    diameters = []

    result = image.copy()

    print()
    print("==========", name, "==========")
    print("width :", width)
    print("height :", height)
    print("center_y :", center_y)
    print("x_start :", x_start)
    print("x_end :", x_end)


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

        diameters.append(
            diameter
        )


        print(
            "x :", x,
            "top :", y_top,
            "bottom :", y_bottom,
            "diameter :", diameter
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


    median_pixel = np.median(
        diameters
    )

    print("측정값 :", diameters)
    print("중앙값 :", median_pixel)

    return (
        edge,
        result,
        median_pixel,
        width,
        height
    )


# -----------------------------
# 3. Day016 기준 이미지 측정
# -----------------------------

edge_016, result_016, pixel_016, width_016, height_016 = measure_diameter(
    image_016,
    "Day016 기준"
)


# -----------------------------
# 4. Day017 검증 이미지 측정
# -----------------------------

edge_017, result_017, pixel_017, width_017, height_017 = measure_diameter(
    image_017,
    "Day017 검증"
)


# -----------------------------
# 5. 비교 결과
# -----------------------------

print()
print("========== 비교 결과 ==========")

print("Day016 ROI 크기 :", width_016, "x", height_016)
print("Day017 ROI 크기 :", width_017, "x", height_017)

print("Day016 외경(pixel) :", pixel_016)
print("Day017 외경(pixel) :", pixel_017)

pixel_difference = pixel_017 - pixel_016

print("Pixel 차이 :", pixel_difference)


# -----------------------------
# 6. 저장
# -----------------------------

output_dir = Path("Day017_1")

output_dir.mkdir(exist_ok=True)


cv2.imwrite(
    str(output_dir / "day016_debug.jpg"),
    result_016
)

cv2.imwrite(
    str(output_dir / "day017_debug.jpg"),
    result_017
)

cv2.imwrite(
    str(output_dir / "day016_edge.jpg"),
    edge_016
)

cv2.imwrite(
    str(output_dir / "day017_edge.jpg"),
    edge_017
)


print()
print("저장 완료")