import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로
# ============================================================

BASE_DIR = Path("Day026-5")
IMAGE_PATH = BASE_DIR / "sample_07.jpg"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(
    "runs/detect/Day023/runs/"
    "measurement_roi_aug-2/weights/best.pt"
)


# ============================================================
# 2. YOLO ROI 검출
# ============================================================

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    imgsz=640,
    verbose=False
)

best_box = None
best_conf = 0.0

for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        confidence = float(box.conf[0])

        if confidence > best_conf:
            best_conf = confidence
            best_box = box


if best_box is None:
    raise RuntimeError(
        "YOLO ROI 검출 실패"
    )


if best_conf < 0.70:
    raise RuntimeError(
        f"YOLO confidence 부족 : {best_conf:.3f}"
    )


# ============================================================
# 3. 이미지
# ============================================================

image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:
    raise FileNotFoundError(
        IMAGE_PATH
    )


image_h, image_w = image.shape[:2]

x1, y1, x2, y2 = (
    best_box.xyxy[0]
    .cpu()
    .numpy()
    .astype(int)
)


x1 = max(0, x1)
y1 = max(0, y1)

x2 = min(image_w, x2)
y2 = min(image_h, y2)


print()
print("========== YOLO ROI ==========")

print("confidence :", best_conf)
print("x1 :", x1)
print("y1 :", y1)
print("x2 :", x2)
print("y2 :", y2)


# ============================================================
# 4. 측정 ROI 확장
# Day026-4와 동일
# ============================================================

box_h = y2 - y1

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
    raise RuntimeError(
        "측정 ROI가 비어 있습니다."
    )


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "expanded_roi.jpg"
    ),
    roi
)


# ============================================================
# 5. 전처리
# ============================================================

gray = cv2.cvtColor(
    roi,
    cv2.COLOR_BGR2GRAY
)

blur = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)

roi_h, roi_w = blur.shape


# ============================================================
# 6. Sub-pixel 보간
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
# 7. Day026-4 방식 Edge 검출
#
# 방향이 맞고
# 충분히 강하며
# 가능한 후보 중 바깥쪽 Pair 선택
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


    # 주변 5 pixel 평균
    profile = np.mean(
        gray[
            :,
            strip_x1:strip_x2
        ],
        axis=1
    ).astype(np.float32)


    # signed gradient
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


    # 외곽 후보가 살아남도록 82 percentile
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


            # 상하 Edge gradient 방향은 반대
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


            # 내부 Edge 제거
            if diameter < h * 0.25:
                continue


            # 지나친 배경 Edge 제거
            if diameter > h * 0.60:
                continue


            strength = (
                abs(top_grad)
                + abs(bottom_grad)
            )


            candidate_pairs.append(
                (
                    top_idx,
                    bottom_idx,
                    diameter,
                    strength
                )
            )


    if len(candidate_pairs) == 0:
        return None


    # ========================================================
    # 충분히 강한 후보만 유지
    # ========================================================

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
        if pair[3] >= strength_limit
    ]


    if len(strong_pairs) == 0:
        strong_pairs = candidate_pairs


    # ========================================================
    # 방향 + 강도 조건을 만족하면서
    # 가장 바깥쪽
    # ========================================================

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


    # ========================================================
    # Sub-pixel 위치
    # ========================================================

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
# 8. 31개 위치에서 Sub-pixel Edge 검출
#
# Day026-4의 21개보다 증가
# ============================================================

x_positions = np.linspace(
    int(roi_w * 0.18),
    int(roi_w * 0.82),
    31
).astype(int)


top_points = []
bottom_points = []
raw_vertical_diameters = []


for x in x_positions:

    result = find_outer_directional_edge(
        blur,
        x
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


    raw_vertical_diameters.append(
        diameter
    )


top_points = np.array(
    top_points,
    dtype=np.float64
)


bottom_points = np.array(
    bottom_points,
    dtype=np.float64
)


raw_vertical_diameters = np.array(
    raw_vertical_diameters,
    dtype=np.float64
)


print()
print(
    "========== Raw Sub-pixel =========="
)


print(
    "유효 측정 수 :",
    len(raw_vertical_diameters)
)


print(
    "세로 외경(pixel) :",
    np.round(
        raw_vertical_diameters,
        3
    )
)


if len(raw_vertical_diameters) < 10:

    raise RuntimeError(
        "유효 측정점이 부족합니다."
    )


# ============================================================
# 9. 1차 MAD 이상값 제거
# ============================================================

raw_median = float(
    np.median(
        raw_vertical_diameters
    )
)


deviation = np.abs(
    raw_vertical_diameters
    - raw_median
)


mad = float(
    np.median(
        deviation
    )
)


if mad > 0:

    robust_sigma = (
        1.4826
        * mad
    )


    threshold = (
        3.0
        * robust_sigma
    )


    valid_mask = (
        deviation
        <= threshold
    )

else:

    valid_mask = np.ones(
        len(raw_vertical_diameters),
        dtype=bool
    )


top_valid = top_points[
    valid_mask
]


bottom_valid = bottom_points[
    valid_mask
]


vertical_valid = raw_vertical_diameters[
    valid_mask
]


print()
print(
    "========== 1차 이상값 제거 =========="
)

print(
    "유효 측정 수 :",
    len(vertical_valid)
)

print(
    "측정값 :",
    np.round(
        vertical_valid,
        3
    )
)


if len(vertical_valid) < 10:

    raise RuntimeError(
        "이상값 제거 후 측정점 부족"
    )


# ============================================================
# 10. 상단 / 하단 Line Fit
#
# 여기서는 외경을 결정하려는 것이 아니라
# 샤프트의 기울기만 안정적으로 계산
# ============================================================

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


slope_difference = abs(
    top_a
    - bottom_a
)


common_slope = (
    top_a
    + bottom_a
) / 2.0


print()
print(
    "========== 기울기 분석 =========="
)

print(
    "상단 기울기 :",
    top_a
)

print(
    "하단 기울기 :",
    bottom_a
)

print(
    "기울기 차이 :",
    slope_difference
)

print(
    "공통 기울기 :",
    common_slope
)


MAX_SLOPE_DIFFERENCE = 0.08


parallel_ok = (
    slope_difference
    <= MAX_SLOPE_DIFFERENCE
)


print(
    "평행성 판정 :",
    "PASS"
    if parallel_ok
    else "FAIL"
)


if not parallel_ok:

    raise RuntimeError(
        "상단/하단 Edge 평행성 불량"
    )


# ============================================================
# 11. 각 X 위치별 실제 수직거리
#
# 핵심 Day026-5
#
# 같은 x에서 측정한 y 차이는 세로거리이고,
# 샤프트가 기울어져 있으므로 실제 외경은
# 공통 기울기에 수직인 거리로 변환
#
# d_perpendicular =
#
# d_vertical / sqrt(1 + slope^2)
# ============================================================

correction_factor = np.sqrt(
    1.0
    + common_slope ** 2
)


perpendicular_distances = (
    vertical_valid
    / correction_factor
)


print()
print(
    "========== 다점 실제 수직거리 =========="
)


print(
    "수직거리(pixel) :",
    np.round(
        perpendicular_distances,
        4
    )
)


print(
    "보정 계수 :",
    correction_factor
)


# ============================================================
# 12. 수직거리 기준 2차 MAD 제거
# ============================================================

distance_median = float(
    np.median(
        perpendicular_distances
    )
)


distance_deviation = np.abs(
    perpendicular_distances
    - distance_median
)


distance_mad = float(
    np.median(
        distance_deviation
    )
)


if distance_mad > 0:

    distance_sigma = (
        1.4826
        * distance_mad
    )


    distance_threshold = (
        3.0
        * distance_sigma
    )


    distance_mask = (
        distance_deviation
        <= distance_threshold
    )

else:

    distance_mask = np.ones(
        len(perpendicular_distances),
        dtype=bool
    )


final_distances = perpendicular_distances[
    distance_mask
]


final_top_points = top_valid[
    distance_mask
]


final_bottom_points = bottom_valid[
    distance_mask
]


print()
print(
    "========== 2차 이상값 제거 =========="
)


print(
    "최종 유효 측정 수 :",
    len(final_distances)
)


print(
    "최종 측정값(pixel) :",
    np.round(
        final_distances,
        4
    )
)


# ============================================================
# 13. 최종 대표 외경
#
# 평균보다 Median을 대표값으로 사용
# ============================================================

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


print()
print(
    "========== Day026-5 최종 =========="
)


print(
    "최종 Median 외경(pixel) :",
    final_median
)


print(
    "최종 평균 외경(pixel) :",
    final_mean
)


print(
    "최종 표준편차(pixel) :",
    final_std
)


# ============================================================
# 14. Day026-4와 비교용 Line 거리
# ============================================================

top_intercepts = (
    final_top_points[:, 1]
    - common_slope
    * final_top_points[:, 0]
)


bottom_intercepts = (
    final_bottom_points[:, 1]
    - common_slope
    * final_bottom_points[:, 0]
)


top_intercept = float(
    np.median(
        top_intercepts
    )
)


bottom_intercept = float(
    np.median(
        bottom_intercepts
    )
)


line_distance = (
    abs(
        bottom_intercept
        - top_intercept
    )
    / correction_factor
)


print()
print(
    "Line 기반 외경(pixel) :",
    line_distance
)


# ============================================================
# 15. 기존 기준값과 비교
#
# 측정 알고리즘에는 사용하지 않음
# ============================================================

REFERENCE_PIXEL = 462.72


difference = (
    final_median
    - REFERENCE_PIXEL
)


print()
print(
    "========== 기존 기준 비교 =========="
)


print(
    "기준값(pixel) :",
    REFERENCE_PIXEL
)


print(
    "Day026-5 측정값(pixel) :",
    final_median
)


print(
    "차이(pixel) :",
    difference
)


# ============================================================
# 16. 시각화
# ============================================================

debug = roi.copy()


# 최종 유효점
for x, y in final_top_points:

    cv2.circle(
        debug,
        (
            int(round(x)),
            int(round(y))
        ),
        4,
        (0, 0, 255),
        -1
    )


for x, y in final_bottom_points:

    cv2.circle(
        debug,
        (
            int(round(x)),
            int(round(y))
        ),
        4,
        (255, 0, 0),
        -1
    )


# ============================================================
# 17. 평행 대표선
# ============================================================

x_start = 0
x_end = roi_w - 1


top_start = int(
    round(
        common_slope
        * x_start
        + top_intercept
    )
)


top_end = int(
    round(
        common_slope
        * x_end
        + top_intercept
    )
)


bottom_start = int(
    round(
        common_slope
        * x_start
        + bottom_intercept
    )
)


bottom_end = int(
    round(
        common_slope
        * x_end
        + bottom_intercept
    )
)


cv2.line(
    debug,
    (x_start, top_start),
    (x_end, top_end),
    (0, 255, 0),
    2
)


cv2.line(
    debug,
    (x_start, bottom_start),
    (x_end, bottom_end),
    (0, 255, 0),
    2
)


# ============================================================
# 18. 여러 실제 측정선 표시
#
# Day026-5에서는 한 개가 아니라
# 유효 측정점들을 표시
# ============================================================

for (
    top_point,
    bottom_point
) in zip(
    final_top_points,
    final_bottom_points
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


    cv2.line(
        debug,
        (x, top_y),
        (x, bottom_y),
        (0, 255, 255),
        1
    )


# 최종 값 표시
cv2.putText(
    debug,
    f"Median: {final_median:.3f} px",
    (5, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2
)


# ============================================================
# 19. 저장
# ============================================================

output_path = (
    OUTPUT_DIR
    / "multipoint_subpixel_median.jpg"
)


cv2.imwrite(
    str(output_path),
    debug
)


print()
print(
    "결과 저장 :",
    output_path
)


print()
print(
    "=============================="
)

print(
    "Day026-5 완료"
)

print(
    "=============================="
)