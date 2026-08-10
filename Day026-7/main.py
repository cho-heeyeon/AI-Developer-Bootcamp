import cv2
import csv
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로
# ============================================================

BASE_DIR = Path("Day026-7")

IMAGE_DIR = BASE_DIR / "images"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = Path(
    "runs/detect/Day023/runs/"
    "measurement_roi_aug-2/weights/best.pt"
)


SAMPLES = [
    "sample_03",
    "sample_04",
    "sample_06",
    "sample_07",
    "sample_N01",
    "sample_N02",
]


# ============================================================
# 2. YOLO 모델
# ============================================================

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# 3. Sub-pixel 함수
# ============================================================

def subpixel_edge(
    abs_gradient,
    index
):

    if index <= 0:
        return float(index)

    if index >= len(abs_gradient) - 1:
        return float(index)


    g1 = float(
        abs_gradient[index - 1]
    )

    g2 = float(
        abs_gradient[index]
    )

    g3 = float(
        abs_gradient[index + 1]
    )


    denominator = (
        g1
        - 2.0 * g2
        + g3
    )


    if abs(denominator) < 1e-12:
        return float(index)


    offset = (
        0.5
        * (g1 - g3)
        / denominator
    )


    offset = np.clip(
        offset,
        -1.0,
        1.0
    )


    return (
        float(index)
        + float(offset)
    )


# ============================================================
# 4. Edge 검출
#
# Day026-5 방식
# ============================================================

def find_outer_directional_edge(
    gray,
    x
):

    strip_x1 = max(
        0,
        x - 2
    )

    strip_x2 = min(
        gray.shape[1],
        x + 3
    )


    profile = np.mean(
        gray[
            :,
            strip_x1:strip_x2
        ],
        axis=1
    ).astype(np.float32)


    gradient = np.gradient(
        profile
    )


    abs_gradient = np.abs(
        gradient
    )


    h = len(profile)


    upper_end = int(
        h * 0.46
    )

    lower_start = int(
        h * 0.54
    )


    upper_abs = abs_gradient[
        :upper_end
    ]


    lower_abs = abs_gradient[
        lower_start:
    ]


    if len(upper_abs) == 0:
        return None

    if len(lower_abs) == 0:
        return None


    upper_threshold = np.percentile(
        upper_abs,
        82
    )


    lower_threshold = np.percentile(
        lower_abs,
        82
    )


    upper_candidates = np.where(
        upper_abs
        >= upper_threshold
    )[0]


    lower_candidates_local = np.where(
        lower_abs
        >= lower_threshold
    )[0]


    lower_candidates = (
        lower_candidates_local
        + lower_start
    )


    if len(upper_candidates) == 0:
        return None

    if len(lower_candidates) == 0:
        return None


    candidate_pairs = []


    for top_idx in upper_candidates:

        top_grad = float(
            gradient[top_idx]
        )


        for bottom_idx in lower_candidates:

            bottom_grad = float(
                gradient[bottom_idx]
            )


            # 상하 Edge 방향 반대
            if (
                top_grad
                * bottom_grad
                >= 0
            ):
                continue


            diameter = (
                bottom_idx
                - top_idx
            )


            if diameter < h * 0.25:
                continue


            if diameter > h * 0.60:
                continue


            strength = (
                abs(top_grad)
                + abs(bottom_grad)
            )


            candidate_pairs.append(
                (
                    int(top_idx),
                    int(bottom_idx),
                    float(diameter),
                    float(strength)
                )
            )


    if len(candidate_pairs) == 0:
        return None


    strengths = np.array(
        [
            pair[3]
            for pair in candidate_pairs
        ]
    )


    strength_limit = (
        np.max(strengths)
        * 0.55
    )


    strong_pairs = [
        pair
        for pair in candidate_pairs
        if pair[3]
        >= strength_limit
    ]


    if len(strong_pairs) == 0:

        strong_pairs = (
            candidate_pairs
        )


    # 가장 바깥쪽
    best_pair = max(
        strong_pairs,
        key=lambda pair: pair[2]
    )


    top_idx = int(
        best_pair[0]
    )


    bottom_idx = int(
        best_pair[1]
    )


    top_sub = subpixel_edge(
        abs_gradient,
        top_idx
    )


    bottom_sub = subpixel_edge(
        abs_gradient,
        bottom_idx
    )


    diameter_sub = (
        bottom_sub
        - top_sub
    )


    return (
        top_sub,
        bottom_sub,
        diameter_sub
    )


# ============================================================
# 5. MAD Mask
# ============================================================

def mad_mask(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )


    median = np.median(
        values
    )


    deviation = np.abs(
        values
        - median
    )


    mad = np.median(
        deviation
    )


    if mad <= 0:

        return np.ones(
            len(values),
            dtype=bool
        )


    sigma = (
        1.4826
        * mad
    )


    threshold = (
        3.0
        * sigma
    )


    return (
        deviation
        <= threshold
    )


# ============================================================
# 6. 샘플 1개 측정 함수
# ============================================================

def measure_sample(
    sample_name
):

    image_path = (
        IMAGE_DIR
        / f"{sample_name}.jpg"
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        return {
            "sample": sample_name,
            "status": "IMAGE_FAIL"
        }


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    predictions = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        verbose=False
    )


    best_box = None
    best_conf = 0.0


    for prediction in predictions:

        if prediction.boxes is None:
            continue


        for box in prediction.boxes:

            conf = float(
                box.conf[0]
            )


            if conf > best_conf:

                best_conf = conf
                best_box = box


    if best_box is None:

        return {
            "sample": sample_name,
            "status": "YOLO_FAIL"
        }


    # --------------------------------------------------------
    # 신뢰도
    # --------------------------------------------------------

    if best_conf < 0.70:

        return {
            "sample": sample_name,
            "status": "LOW_CONF",
            "confidence": best_conf
        }


    image_h, image_w = (
        image.shape[:2]
    )


    x1, y1, x2, y2 = (
        best_box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )


    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        image_w,
        x2
    )

    y2 = min(
        image_h,
        y2
    )


    # --------------------------------------------------------
    # ROI 확장
    # --------------------------------------------------------

    box_h = (
        y2 - y1
    )


    vertical_margin = int(
        box_h * 0.40
    )


    ey1 = max(
        0,
        y1 - vertical_margin
    )


    ey2 = min(
        image_h,
        y2 + vertical_margin
    )


    roi = image[
        ey1:ey2,
        x1:x2
    ].copy()


    if roi.size == 0:

        return {
            "sample": sample_name,
            "status": "ROI_FAIL"
        }


    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    roi_h, roi_w = (
        blur.shape
    )


    # --------------------------------------------------------
    # 31개 x 위치 측정
    # --------------------------------------------------------

    x_positions = np.linspace(
        int(roi_w * 0.18),
        int(roi_w * 0.82),
        31
    ).astype(int)


    top_points = []

    bottom_points = []

    diameters = []


    for x in x_positions:

        result = (
            find_outer_directional_edge(
                blur,
                x
            )
        )


        if result is None:
            continue


        (
            top_y,
            bottom_y,
            diameter
        ) = result


        top_points.append(
            [x, top_y]
        )


        bottom_points.append(
            [x, bottom_y]
        )


        diameters.append(
            diameter
        )


    if len(diameters) < 10:

        return {
            "sample": sample_name,
            "status": "EDGE_FAIL",
            "confidence": best_conf,
            "valid_count": len(
                diameters
            )
        }


    top_points = np.array(
        top_points,
        dtype=np.float64
    )


    bottom_points = np.array(
        bottom_points,
        dtype=np.float64
    )


    diameters = np.array(
        diameters,
        dtype=np.float64
    )


    # --------------------------------------------------------
    # 1차 MAD
    # --------------------------------------------------------

    mask1 = mad_mask(
        diameters
    )


    top_valid = (
        top_points[
            mask1
        ]
    )


    bottom_valid = (
        bottom_points[
            mask1
        ]
    )


    diameter_valid = (
        diameters[
            mask1
        ]
    )


    if len(diameter_valid) < 10:

        return {
            "sample": sample_name,
            "status": "MAD_FAIL",
            "confidence": best_conf
        }


    # --------------------------------------------------------
    # 기울기
    # --------------------------------------------------------

    top_a, top_b = np.polyfit(
        top_valid[:, 0],
        top_valid[:, 1],
        1
    )


    bottom_a, bottom_b = (
        np.polyfit(
            bottom_valid[:, 0],
            bottom_valid[:, 1],
            1
        )
    )


    slope_diff = abs(
        top_a
        - bottom_a
    )


    common_slope = (
        top_a
        + bottom_a
    ) / 2.0


    # --------------------------------------------------------
    # 수직거리
    # --------------------------------------------------------

    factor = np.sqrt(
        1.0
        + common_slope ** 2
    )


    perpendicular = (
        diameter_valid
        / factor
    )


    # --------------------------------------------------------
    # 2차 MAD
    # --------------------------------------------------------

    mask2 = mad_mask(
        perpendicular
    )


    final_distances = (
        perpendicular[
            mask2
        ]
    )


    final_top = (
        top_valid[
            mask2
        ]
    )


    final_bottom = (
        bottom_valid[
            mask2
        ]
    )


    if len(final_distances) < 8:

        return {
            "sample": sample_name,
            "status": "FINAL_FAIL",
            "confidence": best_conf
        }


    final_median = float(
        np.median(
            final_distances
        )
    )


    final_mean = float(
        np.mean(
            final_distances
        )
    )


    final_std = float(
        np.std(
            final_distances
        )
    )


    # --------------------------------------------------------
    # 시각화
    # --------------------------------------------------------

    debug = roi.copy()


    for (
        top_point,
        bottom_point
    ) in zip(
        final_top,
        final_bottom
    ):

        x = int(
            round(
                top_point[0]
            )
        )


        top_y = int(
            round(
                top_point[1]
            )
        )


        bottom_y = int(
            round(
                bottom_point[1]
            )
        )


        cv2.circle(
            debug,
            (x, top_y),
            3,
            (0, 0, 255),
            -1
        )


        cv2.circle(
            debug,
            (x, bottom_y),
            3,
            (255, 0, 0),
            -1
        )


        cv2.line(
            debug,
            (x, top_y),
            (x, bottom_y),
            (0, 255, 255),
            1
        )


    cv2.putText(
        debug,
        f"{sample_name}: "
        f"{final_median:.3f}px",
        (5, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{sample_name}_measurement.jpg"
        ),
        debug
    )


    # --------------------------------------------------------
    # 결과
    # --------------------------------------------------------

    return {

        "sample":
            sample_name,

        "status":
            "PASS",

        "confidence":
            best_conf,

        "valid_count":
            len(final_distances),

        "median_px":
            final_median,

        "mean_px":
            final_mean,

        "std_px":
            final_std,

        "top_slope":
            float(top_a),

        "bottom_slope":
            float(bottom_a),

        "slope_diff":
            float(slope_diff)
    }


# ============================================================
# 7. 6개 일괄 실행
# ============================================================

all_results = []


print()
print(
    "=========================================="
)

print(
    "Day026-7 6개 샘플 일괄 측정"
)

print(
    "=========================================="
)


for sample in SAMPLES:

    result = measure_sample(
        sample
    )


    all_results.append(
        result
    )


    print()
    print(
        "Sample :",
        sample
    )


    print(
        "Status :",
        result[
            "status"
        ]
    )


    if (
        result["status"]
        == "PASS"
    ):

        print(
            "YOLO confidence :",
            result[
                "confidence"
            ]
        )


        print(
            "Median(px) :",
            result[
                "median_px"
            ]
        )


        print(
            "Mean(px) :",
            result[
                "mean_px"
            ]
        )


        print(
            "STD(px) :",
            result[
                "std_px"
            ]
        )


        print(
            "Slope diff :",
            result[
                "slope_diff"
            ]
        )


# ============================================================
# 8. CSV 저장
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "measurement_results.csv"
)


fieldnames = [
    "sample",
    "status",
    "confidence",
    "valid_count",
    "median_px",
    "mean_px",
    "std_px",
    "top_slope",
    "bottom_slope",
    "slope_diff",
]


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()


    for result in all_results:

        row = {
            key: result.get(
                key,
                ""
            )
            for key in fieldnames
        }


        writer.writerow(
            row
        )


# ============================================================
# 9. 전체 결과 요약
# ============================================================

pass_results = [

    r

    for r in all_results

    if r["status"]
    == "PASS"
]


print()
print(
    "=========================================="
)

print(
    "전체 결과 요약"
)

print(
    "=========================================="
)


print(
    "전체 샘플 :",
    len(
        all_results
    )
)


print(
    "PASS :",
    len(
        pass_results
    )
)


print(
    "FAIL :",
    len(
        all_results
    )
    - len(
        pass_results
    )
)


if len(pass_results) > 0:

    std_values = np.array(
        [
            r["std_px"]
            for r in pass_results
        ]
    )


    confidence_values = np.array(
        [
            r["confidence"]
            for r in pass_results
        ]
    )


    slope_values = np.array(
        [
            r["slope_diff"]
            for r in pass_results
        ]
    )


    print()
    print(
        "평균 YOLO confidence :",
        np.mean(
            confidence_values
        )
    )


    print(
        "평균 측정 STD(pixel) :",
        np.mean(
            std_values
        )
    )


    print(
        "최대 측정 STD(pixel) :",
        np.max(
            std_values
        )
    )


    print(
        "평균 slope difference :",
        np.mean(
            slope_values
        )
    )


print()
print(
    "CSV 저장 :",
    csv_path
)


print()
print(
    "=========================================="
)

print(
    "Day026-7 완료"
)

print(
    "=========================================="
)