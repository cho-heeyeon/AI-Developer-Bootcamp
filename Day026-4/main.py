import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path("Day026-4")

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
# 3. 원본 이미지
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

print(
    "confidence :",
    best_conf
)

print("x1 :", x1)
print("y1 :", y1)
print("x2 :", x2)
print("y2 :", y2)


# ============================================================
# 4. 측정 ROI
#
# Day025에서 잘 동작한 확장 비율 유지
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
# 5. GRAY / Blur
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
# 6. Sub-pixel Edge
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
# 7. 방향 + 바깥 위치 Edge 선택
# ============================================================

def find_outer_directional_edge(
    gray,
    x
):

    # -----------------------------------------------
    # 한 줄보다 주변 평균 사용
    # -----------------------------------------------

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


    # signed gradient
    gradient = np.gradient(
        profile
    )


    abs_gradient = np.abs(
        gradient
    )


    h = len(profile)

    center = h // 2


    # -----------------------------------------------
    # 중앙 반사영역을 제외한 상/하 검색 영역
    # -----------------------------------------------

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


    # -----------------------------------------------
    # 강한 후보만 남김
    #
    # Day026-3보다 threshold를 약간 낮춰
    # 바깥 Edge 후보도 살아남게 함
    # -----------------------------------------------

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


    # ========================================================
    # 후보 Pair 평가
    # ========================================================

    candidate_pairs = []


    for top_idx in upper_candidates:

        top_grad = float(
            gradient[top_idx]
        )


        for bottom_idx in lower_candidates:

            bottom_grad = float(
                gradient[bottom_idx]
            )


            # -----------------------------------------------
            # 실제 상/하 경계는 Gradient 방향이 반대
            # -----------------------------------------------

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


            # 너무 작은 내부 Edge 제거
            if diameter < h * 0.25:
                continue


            # ROI 거의 전체를 잡는 배경 Edge 제거
            if diameter > h * 0.60:
                continue


            # -----------------------------------------------
            # Edge 강도
            # -----------------------------------------------

            strength = (
                abs(top_grad)
                + abs(bottom_grad)
            )


            # -----------------------------------------------
            # 바깥쪽 정도
            #
            # top은 작을수록 바깥쪽
            # bottom은 클수록 바깥쪽
            # -----------------------------------------------

            outer_distance = (
                diameter
            )


            candidate_pairs.append(
                (
                    top_idx,
                    bottom_idx,
                    diameter,
                    strength,
                    outer_distance
                )
            )


    if len(candidate_pairs) == 0:

        return None


    # ========================================================
    # 8. 강도 필터
    #
    # 아주 약한 바깥 Edge를 선택하지 않도록
    # 강한 후보군만 우선 남김
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
    # 9. 핵심 선택
    #
    # "방향이 맞고 충분히 강한 후보들 중
    #  가장 바깥쪽 Pair"
    #
    # 즉 diameter가 가장 큰 Pair 선택
    # ========================================================

    best_pair = max(
        strong_pairs,
        key=lambda pair: pair[4]
    )


    (
        top_idx,
        bottom_idx,
        diameter_integer,
        strength,
        outer_distance
    ) = best_pair


    # ========================================================
    # 10. Sub-pixel
    # ========================================================

    top_sub = subpixel_edge(
        abs_gradient,
        int(top_idx)
    )


    bottom_sub = subpixel_edge(
        abs_gradient,
        int(bottom_idx)
    )


    diameter_sub = (
        bottom_sub
        - top_sub
    )


    return (
        top_sub,
        bottom_sub,
        diameter_sub,
        float(gradient[top_idx]),
        float(gradient[bottom_idx])
    )


# ============================================================
# 11. 여러 위치 측정
# ============================================================

x_positions = np.linspace(
    int(roi_w * 0.18),
    int(roi_w * 0.82),
    21
).astype(int)


top_points = []

bottom_points = []

diameters = []


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
        diameter,
        top_grad,
        bottom_grad
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


print()
print(
    "========== Day026-4 Raw =========="
)


print(
    "유효 점 개수 :",
    len(diameters)
)


print(
    "Raw 외경(pixel) :",
    np.round(
        diameters,
        3
    )
)


if len(diameters) < 7:

    raise RuntimeError(
        "유효 Edge 점 부족"
    )


# ============================================================
# 12. MAD 이상값 제거
# ============================================================

raw_median = float(
    np.median(
        diameters
    )
)


deviation = np.abs(
    diameters
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
        len(diameters),
        dtype=bool
    )


top_valid = top_points[
    valid_mask
]


bottom_valid = bottom_points[
    valid_mask
]


diameter_valid = diameters[
    valid_mask
]


print()
print(
    "========== 이상값 제거 =========="
)


print(
    "유효 외경(pixel) :",
    np.round(
        diameter_valid,
        3
    )
)


print(
    "Raw 중앙값(pixel) :",
    np.median(
        diameter_valid
    )
)


print(
    "Raw 평균(pixel) :",
    np.mean(
        diameter_valid
    )
)


print(
    "Raw 표준편차(pixel) :",
    np.std(
        diameter_valid
    )
)


if len(diameter_valid) < 7:

    raise RuntimeError(
        "이상값 제거 후 유효점 부족"
    )


# ============================================================
# 13. 상 / 하단 Line Fitting
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


print()
print(
    "========== Line Fitting =========="
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


# ============================================================
# 14. 평행성 검사
# ============================================================

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


# ============================================================
# 15. 공통 기울기
# ============================================================

common_slope = (
    top_a
    + bottom_a
) / 2.0


top_intercepts = (
    top_valid[:, 1]
    - common_slope
    * top_valid[:, 0]
)


bottom_intercepts = (
    bottom_valid[:, 1]
    - common_slope
    * bottom_valid[:, 0]
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


# ============================================================
# 16. 실제 평행선 수직거리
#
# y = mx + b인 두 평행선의
# 수직 거리는:
#
# |b2 - b1| / sqrt(1 + m^2)
#
# Day026-1에서는 y 차이를 사용했지만
# 이번에는 샤프트 기울기까지 반영한
# 실제 perpendicular distance를 계산합니다.
# ============================================================

vertical_distance = (
    bottom_intercept
    - top_intercept
)


perpendicular_distance = (
    abs(vertical_distance)
    / np.sqrt(
        1.0
        + common_slope ** 2
    )
)


print()
print(
    "========== Day026-4 최종 =========="
)


print(
    "공통 기울기 :",
    common_slope
)


print(
    "세로방향 거리(pixel) :",
    vertical_distance
)


print(
    "수직 외경(pixel) :",
    perpendicular_distance
)


# ============================================================
# 17. 기존 기준값과 비교
#
# 선택에는 사용하지 않음
# ============================================================

REFERENCE_PIXEL = 462.72


difference = (
    perpendicular_distance
    - REFERENCE_PIXEL
)


print()
print(
    "========== 기준 비교 =========="
)


print(
    "기준값(pixel) :",
    REFERENCE_PIXEL
)


print(
    "측정값(pixel) :",
    perpendicular_distance
)


print(
    "차이(pixel) :",
    difference
)


# ============================================================
# 18. 시각화
# ============================================================

debug = roi.copy()


# 실제 검출 점
for x, y in top_valid:

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


for x, y in bottom_valid:

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


# fitted line
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
    3
)


cv2.line(
    debug,
    (x_start, bottom_start),
    (x_end, bottom_end),
    (0, 255, 0),
    3
)


# 중앙 표시선
cx = roi_w // 2


cy_top = int(
    round(
        common_slope
        * cx
        + top_intercept
    )
)


cy_bottom = int(
    round(
        common_slope
        * cx
        + bottom_intercept
    )
)


cv2.line(
    debug,
    (
        cx,
        cy_top
    ),
    (
        cx,
        cy_bottom
    ),
    (0, 255, 255),
    3
)


cv2.putText(
    debug,
    f"{perpendicular_distance:.3f} px",
    (5, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 255),
    2
)


output_path = (
    OUTPUT_DIR
    / "outer_directional_edge.jpg"
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
print("==============================")
print("Day026-4 완료")
print("==============================")