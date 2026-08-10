import cv2
import csv
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로
# ============================================================

BASE_DIR = Path("Day026-8")

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
# 2. 판정 기준
# ============================================================

YOLO_PASS_CONF = 0.70
YOLO_WARNING_CONF = 0.60

# 측정 반복성
STD_PASS = 1.00
STD_WARNING = 3.00

# 상하 Edge 평행성
SLOPE_PASS = 0.08
SLOPE_WARNING = 0.15


# ============================================================
# 3. YOLO 모델
# ============================================================

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# 4. Sub-pixel
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
# 5. 외곽 Edge 검출
#
# Day026-7과 같은 방식
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


    upper_abs = (
        abs_gradient[:upper_end]
    )

    lower_abs = (
        abs_gradient[lower_start:]
    )


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


            # 상하 Edge 방향은 반대
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


            # 너무 작은 내부 Edge 제외
            if diameter < h * 0.25:
                continue


            # 지나치게 큰 배경 Edge 제외
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
            for pair
            in candidate_pairs
        ]
    )


    strength_limit = (
        np.max(strengths)
        * 0.55
    )


    strong_pairs = [
        pair
        for pair
        in candidate_pairs
        if pair[3]
        >= strength_limit
    ]


    if len(strong_pairs) == 0:

        strong_pairs = (
            candidate_pairs
        )


    # 충분히 강한 후보 중
    # 가장 바깥쪽 Edge
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
# 6. MAD 이상값 제거
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


    robust_sigma = (
        1.4826
        * mad
    )


    threshold = (
        3.0
        * robust_sigma
    )


    return (
        deviation
        <= threshold
    )


# ============================================================
# 7. YOLO 상태 판정
# ============================================================

def get_yolo_status(
    confidence
):

    if confidence >= YOLO_PASS_CONF:

        return "PASS"


    if confidence >= YOLO_WARNING_CONF:

        return "WARNING"


    return "FAIL"


# ============================================================
# 8. 측정 품질 판정
# ============================================================

def get_measurement_status(
    std_px,
    slope_diff
):

    if (
        std_px <= STD_PASS
        and
        slope_diff <= SLOPE_PASS
    ):

        return "PASS"


    if (
        std_px <= STD_WARNING
        and
        slope_diff <= SLOPE_WARNING
    ):

        return "WARNING"


    return "FAIL"


# ============================================================
# 9. 최종 종합 판정
# ============================================================

def get_final_status(
    yolo_status,
    measurement_status
):

    # YOLO 자체가 매우 낮으면
    # 전체 FAIL
    if yolo_status == "FAIL":

        return "FAIL"


    # 측정엔진이 FAIL이면
    # 전체 FAIL
    if measurement_status == "FAIL":

        return "FAIL"


    # 둘 다 PASS
    if (
        yolo_status == "PASS"
        and
        measurement_status == "PASS"
    ):

        return "PASS"


    # 나머지는 경고
    return "WARNING"


# ============================================================
# 10. 샘플 1개 측정
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
            "sample":
                sample_name,

            "final_status":
                "FAIL",

            "reason":
                "IMAGE_FAIL"
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
            "sample":
                sample_name,

            "final_status":
                "FAIL",

            "reason":
                "YOLO_NO_DETECTION"
        }


    # --------------------------------------------------------
    # YOLO 상태
    # --------------------------------------------------------

    yolo_status = (
        get_yolo_status(
            best_conf
        )
    )


    # 중요:
    # WARNING이어도 측정 계속 진행
    #
    # FAIL(<0.60)도 분석 목적상
    # 좌표가 있으면 측정을 시도하도록 함


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
    # ROI
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
            "sample":
                sample_name,

            "confidence":
                best_conf,

            "yolo_status":
                yolo_status,

            "final_status":
                "FAIL",

            "reason":
                "ROI_FAIL"
        }


    # --------------------------------------------------------
    # OpenCV
    # --------------------------------------------------------

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
    # 31개 위치 측정
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

        edge_result = (
            find_outer_directional_edge(
                blur,
                x
            )
        )


        if edge_result is None:
            continue


        (
            top_y,
            bottom_y,
            diameter
        ) = edge_result


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
            "sample":
                sample_name,

            "confidence":
                best_conf,

            "yolo_status":
                yolo_status,

            "valid_count":
                len(diameters),

            "measurement_status":
                "FAIL",

            "final_status":
                "FAIL",

            "reason":
                "EDGE_POINTS_LOW"
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
            "sample":
                sample_name,

            "confidence":
                best_conf,

            "yolo_status":
                yolo_status,

            "measurement_status":
                "FAIL",

            "final_status":
                "FAIL",

            "reason":
                "MAD_POINTS_LOW"
        }


    # --------------------------------------------------------
    # Line fitting → 기울기 계산
    # --------------------------------------------------------

    top_a, top_b = np.polyfit(
        top_valid[:, 0],
        top_valid[:, 1],
        1
    )


    bottom_a, bottom_b = np.polyfit(
        bottom_valid[:, 0],
        bottom_valid[:, 1],
        1
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
    # 실제 수직거리
    # --------------------------------------------------------

    correction_factor = np.sqrt(
        1.0
        + common_slope ** 2
    )


    perpendicular = (
        diameter_valid
        / correction_factor
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
            "sample":
                sample_name,

            "confidence":
                best_conf,

            "yolo_status":
                yolo_status,

            "measurement_status":
                "FAIL",

            "final_status":
                "FAIL",

            "reason":
                "FINAL_POINTS_LOW"
        }


    # --------------------------------------------------------
    # 최종 통계
    # --------------------------------------------------------

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


    measurement_status = (
        get_measurement_status(
            final_std,
            slope_diff
        )
    )


    final_status = (
        get_final_status(
            yolo_status,
            measurement_status
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


    # 결과 문자열
    text1 = (
        f"{sample_name} "
        f"{final_median:.3f}px"
    )


    text2 = (
        f"YOLO:{yolo_status} "
        f"MEASURE:{measurement_status}"
    )


    cv2.putText(
        debug,
        text1,
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    cv2.putText(
        debug,
        text2,
        (5, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1
    )


    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{sample_name}_measurement.jpg"
        ),
        debug
    )


    # --------------------------------------------------------
    # 결과 반환
    # --------------------------------------------------------

    return {

        "sample":
            sample_name,

        "confidence":
            best_conf,

        "yolo_status":
            yolo_status,

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
            float(slope_diff),

        "measurement_status":
            measurement_status,

        "final_status":
            final_status,

        "reason":
            ""
    }


# ============================================================
# 11. 6개 전체 실행
# ============================================================

all_results = []


print()
print(
    "============================================"
)

print(
    "Day026-8 전체 측정 품질 검증"
)

print(
    "============================================"
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
        "--------------------------------"
    )

    print(
        "Sample :",
        sample
    )


    print(
        "YOLO confidence :",
        result.get(
            "confidence",
            "-"
        )
    )


    print(
        "YOLO 판정 :",
        result.get(
            "yolo_status",
            "-"
        )
    )


    print(
        "Median(px) :",
        result.get(
            "median_px",
            "-"
        )
    )


    print(
        "STD(px) :",
        result.get(
            "std_px",
            "-"
        )
    )


    print(
        "Slope diff :",
        result.get(
            "slope_diff",
            "-"
        )
    )


    print(
        "측정 품질 :",
        result.get(
            "measurement_status",
            "-"
        )
    )


    print(
        "최종 판정 :",
        result.get(
            "final_status",
            "FAIL"
        )
    )


# ============================================================
# 12. CSV 저장
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "measurement_quality_results.csv"
)


fieldnames = [
    "sample",
    "confidence",
    "yolo_status",
    "valid_count",
    "median_px",
    "mean_px",
    "std_px",
    "top_slope",
    "bottom_slope",
    "slope_diff",
    "measurement_status",
    "final_status",
    "reason",
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

            key:
                result.get(
                    key,
                    ""
                )

            for key
            in fieldnames
        }


        writer.writerow(
            row
        )


# ============================================================
# 13. 최종 통계
# ============================================================

pass_count = sum(
    1
    for result
    in all_results
    if result.get(
        "final_status"
    ) == "PASS"
)


warning_count = sum(
    1
    for result
    in all_results
    if result.get(
        "final_status"
    ) == "WARNING"
)


fail_count = (
    len(all_results)
    - pass_count
    - warning_count
)


measured_results = [

    result

    for result
    in all_results

    if "median_px"
    in result
]


print()
print(
    "============================================"
)

print(
    "Day026-8 전체 결과"
)

print(
    "============================================"
)


print(
    "전체 샘플 :",
    len(all_results)
)


print(
    "PASS :",
    pass_count
)


print(
    "WARNING :",
    warning_count
)


print(
    "FAIL :",
    fail_count
)


print(
    "외경 측정 완료 :",
    len(
        measured_results
    ),
    "/",
    len(
        all_results
    )
)


if len(measured_results) > 0:

    std_values = np.array(
        [
            result[
                "std_px"
            ]
            for result
            in measured_results
        ]
    )


    confidence_values = np.array(
        [
            result[
                "confidence"
            ]
            for result
            in measured_results
        ]
    )


    slope_values = np.array(
        [
            result[
                "slope_diff"
            ]
            for result
            in measured_results
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
    "============================================"
)

print(
    "Day026-8 완료"
)

print(
    "============================================"
)