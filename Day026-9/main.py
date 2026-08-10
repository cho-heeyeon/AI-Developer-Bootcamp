import cv2
import csv
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로
# ============================================================

BASE_DIR = Path("Day026-9")

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
# 2. YOLO
# ============================================================

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# 3. Sub-pixel
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
# 4. Day026-8과 동일한 Edge 검출
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
        abs_gradient[
            :upper_end
        ]
    )


    lower_abs = (
        abs_gradient[
            lower_start:
        ]
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


    candidate_pairs = []


    for top_idx in upper_candidates:

        top_grad = float(
            gradient[top_idx]
        )


        for bottom_idx in lower_candidates:

            bottom_grad = float(
                gradient[bottom_idx]
            )


            # 실제 상·하 외곽은
            # gradient 방향이 반대여야 함
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
            p[3]
            for p in candidate_pairs
        ]
    )


    strength_limit = (
        np.max(strengths)
        * 0.55
    )


    strong_pairs = [
        p
        for p in candidate_pairs
        if p[3]
        >= strength_limit
    ]


    if len(strong_pairs) == 0:

        strong_pairs = (
            candidate_pairs
        )


    # 가장 바깥쪽 후보
    best_pair = max(
        strong_pairs,
        key=lambda p: p[2]
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


    diameter = (
        bottom_sub
        - top_sub
    )


    return {
        "top_y":
            top_sub,

        "bottom_y":
            bottom_sub,

        "diameter":
            diameter,

        "top_integer":
            top_idx,

        "bottom_integer":
            bottom_idx,

        "strength":
            best_pair[3]
    }


# ============================================================
# 5. 한 샘플 분석
# ============================================================

def analyze_sample(
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

        print(
            sample_name,
            ": 이미지 읽기 실패"
        )

        return []


    # ========================================================
    # YOLO
    # ========================================================

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

        print(
            sample_name,
            ": YOLO 검출 실패"
        )

        return []


    # ========================================================
    # ROI
    # ========================================================

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

        print(
            sample_name,
            ": ROI 실패"
        )

        return []


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


    # ========================================================
    # 31개 측정
    # ========================================================

    x_positions = np.linspace(
        int(roi_w * 0.18),
        int(roi_w * 0.82),
        31
    ).astype(int)


    records = []


    for index, x in enumerate(
        x_positions,
        start=1
    ):

        edge = (
            find_outer_directional_edge(
                blur,
                x
            )
        )


        if edge is None:

            records.append(
                {
                    "sample":
                        sample_name,

                    "point":
                        index,

                    "x":
                        int(x),

                    "top_y":
                        "",

                    "bottom_y":
                        "",

                    "diameter":
                        "",

                    "deviation":
                        "",

                    "outlier":
                        "EDGE_FAIL",

                    "confidence":
                        best_conf
                }
            )

            continue


        records.append(
            {
                "sample":
                    sample_name,

                "point":
                    index,

                "x":
                    int(x),

                "top_y":
                    float(
                        edge["top_y"]
                    ),

                "bottom_y":
                    float(
                        edge["bottom_y"]
                    ),

                "diameter":
                    float(
                        edge["diameter"]
                    ),

                "deviation":
                    0.0,

                "outlier":
                    "",

                "confidence":
                    best_conf
            }
        )


    # ========================================================
    # 정상 측정값만 가져오기
    # ========================================================

    valid_records = [

        r

        for r in records

        if isinstance(
            r["diameter"],
            float
        )
    ]


    if len(valid_records) < 5:

        print(
            sample_name,
            ": 측정점 부족"
        )

        return records


    diameters = np.array(
        [
            r["diameter"]
            for r
            in valid_records
        ],
        dtype=np.float64
    )


    # ========================================================
    # Robust 기준
    #
    # Median + MAD
    # ========================================================

    median = float(
        np.median(
            diameters
        )
    )


    deviations = np.abs(
        diameters
        - median
    )


    mad = float(
        np.median(
            deviations
        )
    )


    robust_sigma = (
        1.4826
        * mad
    )


    # MAD가 너무 작으면
    # 최소 1 pixel 허용
    outlier_threshold = max(
        3.0 * robust_sigma,
        1.0
    )


    # ========================================================
    # Outlier 판정
    # ========================================================

    for r in valid_records:

        deviation = abs(
            r["diameter"]
            - median
        )


        r["deviation"] = (
            deviation
        )


        if (
            deviation
            > outlier_threshold
        ):

            r["outlier"] = (
                "OUTLIER"
            )

        else:

            r["outlier"] = (
                "NORMAL"
            )


    # ========================================================
    # 통계
    # ========================================================

    normal_values = np.array(
        [
            r["diameter"]
            for r in valid_records
            if r["outlier"]
            == "NORMAL"
        ],
        dtype=np.float64
    )


    outlier_records = [

        r

        for r in valid_records

        if r["outlier"]
        == "OUTLIER"
    ]


    print()
    print(
        "========================================"
    )

    print(
        "Sample :",
        sample_name
    )

    print(
        "YOLO confidence :",
        best_conf
    )

    print(
        "Raw Median(px) :",
        median
    )

    print(
        "Raw STD(px) :",
        np.std(
            diameters
        )
    )

    print(
        "MAD :",
        mad
    )

    print(
        "Outlier 기준(pixel) :",
        outlier_threshold
    )

    print(
        "정상 측정선 :",
        len(
            normal_values
        )
    )

    print(
        "Outlier 측정선 :",
        len(
            outlier_records
        )
    )


    if len(normal_values) > 0:

        print(
            "Outlier 제거 후 Median(px) :",
            np.median(
                normal_values
            )
        )

        print(
            "Outlier 제거 후 STD(px) :",
            np.std(
                normal_values
            )
        )


    # ========================================================
    # 각 측정 위치 출력
    # ========================================================

    print()
    print(
        "---------- x 위치별 측정 ----------"
    )


    for r in valid_records:

        marker = (
            " <<< OUTLIER"
            if r["outlier"]
            == "OUTLIER"
            else ""
        )


        print(
            f"#{r['point']:02d}",
            f"x={r['x']:4d}",
            f"top={r['top_y']:8.2f}",
            f"bottom={r['bottom_y']:8.2f}",
            f"diameter={r['diameter']:8.2f}",
            f"dev={r['deviation']:7.2f}",
            marker
        )


    # ========================================================
    # 시각화
    # ========================================================

    debug = roi.copy()


    for r in valid_records:

        x = int(
            r["x"]
        )


        top_y = int(
            round(
                r["top_y"]
            )
        )


        bottom_y = int(
            round(
                r["bottom_y"]
            )
        )


        if (
            r["outlier"]
            == "OUTLIER"
        ):

            # 빨강 = 이상 측정선
            line_color = (
                0,
                0,
                255
            )

            thickness = 3

        else:

            # 초록 = 정상 측정선
            line_color = (
                0,
                255,
                0
            )

            thickness = 1


        cv2.line(
            debug,
            (x, top_y),
            (x, bottom_y),
            line_color,
            thickness
        )


        cv2.circle(
            debug,
            (x, top_y),
            3,
            line_color,
            -1
        )


        cv2.circle(
            debug,
            (x, bottom_y),
            3,
            line_color,
            -1
        )


    # ========================================================
    # 결과 텍스트
    # ========================================================

    cv2.putText(
        debug,
        f"{sample_name}",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )


    cv2.putText(
        debug,
        f"Median={median:.2f}px",
        (5, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1
    )


    cv2.putText(
        debug,
        (
            f"Outlier="
            f"{len(outlier_records)}"
        ),
        (5, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1
    )


    # ========================================================
    # 저장
    # ========================================================

    output_path = (
        OUTPUT_DIR
        / f"{sample_name}_outlier_debug.jpg"
    )


    cv2.imwrite(
        str(output_path),
        debug
    )


    return records


# ============================================================
# 6. 6개 전체 분석
# ============================================================

all_records = []


print()
print(
    "========================================"
)

print(
    "Day026-9 Edge Outlier 원인 추적"
)

print(
    "========================================"
)


for sample in SAMPLES:

    records = analyze_sample(
        sample
    )


    all_records.extend(
        records
    )


# ============================================================
# 7. 상세 CSV
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "edge_outlier_details.csv"
)


fieldnames = [
    "sample",
    "point",
    "x",
    "top_y",
    "bottom_y",
    "diameter",
    "deviation",
    "outlier",
    "confidence",
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


    for record in all_records:

        writer.writerow(
            record
        )


# ============================================================
# 8. 전체 Outlier 요약
# ============================================================

print()
print(
    "========================================"
)

print(
    "전체 Outlier 요약"
)

print(
    "========================================"
)


for sample in SAMPLES:

    sample_records = [

        r

        for r
        in all_records

        if (
            r["sample"]
            == sample
            and
            r["outlier"]
            in [
                "NORMAL",
                "OUTLIER"
            ]
        )
    ]


    normal_count = sum(
        1

        for r
        in sample_records

        if r["outlier"]
        == "NORMAL"
    )


    outlier_count = sum(
        1

        for r
        in sample_records

        if r["outlier"]
        == "OUTLIER"
    )


    print(
        sample,
        "| 정상 :",
        normal_count,
        "| Outlier :",
        outlier_count
    )


print()
print(
    "상세 CSV 저장 :",
    csv_path
)


print()
print(
    "========================================"
)

print(
    "Day026-9 완료"
)

print(
    "========================================"
)