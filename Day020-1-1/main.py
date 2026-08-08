import cv2
import numpy as np
from pathlib import Path


# ---------------------------------
# 1. 폴더 / 샘플 설정
# ---------------------------------

base_dir = Path("Day020-1-1")

samples = {
    "03": "sample_03_roi.jpg",
    "04": "sample_04_roi.jpg",
    "06": "sample_06_roi.jpg",
    "07": "sample_07_roi.jpg",
    "N01": "sample_N01_roi.jpg",
    "N02": "sample_N02_roi.jpg",
}


# ---------------------------------
# 2. Sub-pixel 보간 함수
# ---------------------------------

def subpixel_peak(profile, index):

    if index <= 0 or index >= len(profile) - 1:
        return float(index)

    y1 = profile[index - 1]
    y2 = profile[index]
    y3 = profile[index + 1]

    denominator = y1 - 2 * y2 + y3

    if denominator == 0:
        return float(index)

    offset = 0.5 * (y1 - y3) / denominator

    return float(index + offset)


# ---------------------------------
# 3. 한 이미지 측정 함수
# ---------------------------------

def measure_sample(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다 : {image_path}"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    gradient_y = cv2.Sobel(
        blur,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_abs = np.abs(
        gradient_y
    )

    height, width = gray.shape

    center_y = height // 2

    # 중앙 50% 구간에서 7곳 측정
    x_start = int(width * 0.25)
    x_end = int(width * 0.75)

    x_positions = np.linspace(
        x_start,
        x_end,
        7,
        dtype=int
    )

    top_values = []
    bottom_values = []
    diameters = []

    result = image.copy()

    print()
    print("x | top_sub | bottom_sub | diameter")
    print("------------------------------------------")

    for x in x_positions:

        x1 = max(0, x - 2)
        x2 = min(width, x + 3)

        strip = gradient_abs[:, x1:x2]

        profile = np.mean(
            strip,
            axis=1
        )

        # -------------------------
        # 상단 Edge
        # -------------------------

        top_profile = profile[:center_y]

        top_index = np.argmax(
            top_profile
        )

        y_top = subpixel_peak(
            top_profile,
            top_index
        )

        # -------------------------
        # 하단 Edge
        # -------------------------

        bottom_profile = profile[center_y:]

        bottom_index_local = np.argmax(
            bottom_profile
        )

        bottom_index = (
            center_y
            + bottom_index_local
        )

        y_bottom = subpixel_peak(
            profile,
            bottom_index
        )

        # -------------------------
        # 외경
        # -------------------------

        diameter = y_bottom - y_top

        top_values.append(y_top)
        bottom_values.append(y_bottom)
        diameters.append(diameter)

        print(
            f"{x:4d} | "
            f"{y_top:9.3f} | "
            f"{y_bottom:12.3f} | "
            f"{diameter:8.3f}"
        )

        # -------------------------
        # 이미지 표시
        # -------------------------

        y_top_draw = int(round(y_top))
        y_bottom_draw = int(round(y_bottom))

        # 상단 Edge - 빨강
        cv2.circle(
            result,
            (x, y_top_draw),
            5,
            (0, 0, 255),
            -1
        )

        # 하단 Edge - 파랑
        cv2.circle(
            result,
            (x, y_bottom_draw),
            5,
            (255, 0, 0),
            -1
        )

        # 측정선 - 초록
        cv2.line(
            result,
            (x, y_top_draw),
            (x, y_bottom_draw),
            (0, 255, 0),
            2
        )

    # ---------------------------------
    # 대표값 / 표준편차
    # ---------------------------------

    median_px = np.median(diameters)
    mean_px = np.mean(diameters)
    std_px = np.std(diameters)

    top_std = np.std(top_values)
    bottom_std = np.std(bottom_values)

    return (
        result,
        median_px,
        mean_px,
        std_px,
        top_std,
        bottom_std
    )


# ---------------------------------
# 4. 모든 샘플 실행
# ---------------------------------

print()
print("========== Day020-1-1 Edge 디버깅 ==========")

summary = {}

for name, filename in samples.items():

    print()
    print("======================================")
    print("Sample :", name)
    print("======================================")

    image_path = base_dir / filename

    (
        result,
        median_px,
        mean_px,
        std_px,
        top_std,
        bottom_std
    ) = measure_sample(image_path)

    summary[name] = {
        "median": median_px,
        "mean": mean_px,
        "std": std_px,
        "top_std": top_std,
        "bottom_std": bottom_std
    }

    # 결과 이미지 저장
    output_path = (
        base_dir
        / f"{name}_measure.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result
    )

    print()
    print("중앙값 :", median_px)
    print("평균 :", mean_px)
    print("외경 표준편차 :", std_px)
    print("상단 Edge 표준편차 :", top_std)
    print("하단 Edge 표준편차 :", bottom_std)

    print(
        "결과 이미지 저장 :",
        output_path
    )


# ---------------------------------
# 5. 전체 요약
# ---------------------------------

print()
print("========== 전체 결과 요약 ==========")

for name, values in summary.items():

    print(
        f"{name:>3} | "
        f"median={values['median']:.3f} px | "
        f"std={values['std']:.3f} px | "
        f"top_std={values['top_std']:.3f} | "
        f"bottom_std={values['bottom_std']:.3f}"
    )


print()
print("Edge 디버깅 완료")