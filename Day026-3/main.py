import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로
# ============================================================

BASE_DIR = Path("Day026-3")

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

        conf = float(box.conf[0])

        if conf > best_conf:
            best_conf = conf
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
# 4. Day025 기준 ROI 확장
# ============================================================

box_h = y2 - y1

margin_y = int(
    box_h * 0.40
)

ey1 = max(
    0,
    y1 - margin_y
)

ey2 = min(
    image_h,
    y2 + margin_y
)


roi = image[
    ey1:ey2,
    x1:x2
].copy()


if roi.size == 0:
    raise RuntimeError(
        "ROI가 비어 있습니다."
    )


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "expanded_roi.jpg"
    ),
    roi
)


# ============================================================
# 5. Gray + Blur
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

def subpixel_extremum(
    gradient,
    index
):

    if index <= 0:
        return float(index)

    if index >= len(gradient) - 1:
        return float(index)


    g1 = float(
        gradient[index - 1]
    )

    g2 = float(
        gradient[index]
    )

    g3 = float(
        gradient[index + 1]
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


    return float(
        index + offset
    )


# ============================================================
# 7. Edge 후보 선택 함수
# ============================================================

def find_directional_outer_edges(
    gray,
    x
):

    # 한 column만 쓰지 않고 주변 5pixel 평균
    x1_local = max(
        0,
        x - 2
    )

    x2_local = min(
        gray.shape[1],
        x + 3
    )


    profile = np.mean(
        gray[:, x1_local:x2_local],
        axis=1
    ).astype(np.float32)


    # signed gradient
    gradient = np.gradient(
        profile
    )


    h = len(profile)

    center = h // 2


    # 중앙 반사영역을 피해서
    # 위/아래 후보 영역 분리
    upper_end = int(
        h * 0.46
    )

    lower_start = int(
        h * 0.54
    )


    upper_gradient = gradient[
        :upper_end
    ]

    lower_gradient = gradient[
        lower_start:
    ]


    if len(upper_gradient) == 0:
        return None

    if len(lower_gradient) == 0:
        return None


    # ========================================================
    # Gradient 방향 자동 판별
    #
    # 상단 외곽과 하단 외곽은
    # 서로 반대 sign이어야 함
    # ========================================================

    upper_abs = np.abs(
        upper_gradient
    )

    lower_abs = np.abs(
        lower_gradient
    )


    # 각 영역에서 강한 후보 여러 개
    upper_threshold = np.percentile(
        upper_abs,
        90
    )

    lower_threshold = np.percentile(
        lower_abs,
        90
    )


    upper_candidates = np.where(
        upper_abs >= upper_threshold
    )[0]

    lower_candidates_local = np.where(
        lower_abs >= lower_threshold
    )[0]


    if len(upper_candidates) == 0:
        return None

    if len(lower_candidates_local) == 0:
        return None


    lower_candidates = (
        lower_candidates_local
        + lower_start
    )


    # ========================================================
    # 모든 상단/하단 후보 조합 평가
    # ========================================================

    best_score = -1.0

    best_top = None
    best_bottom = None


    for top_idx in upper_candidates:

        top_gradient_value = float(
            gradient[top_idx]
        )


        for bottom_idx in lower_candidates:

            bottom_gradient_value = float(
                gradient[bottom_idx]
            )


            # -----------------------------------------------
            # 방향이 같으면 내부 반사 edge일 가능성이 높음
            # -----------------------------------------------

            if (
                top_gradient_value
                * bottom_gradient_value
                >= 0
            ):
                continue


            diameter = (
                bottom_idx
                - top_idx
            )


            # 너무 작은 내부 영역 배제
            if diameter < h * 0.25:
                continue


            # 너무 ROI 전체를 먹는 경우 배제
            if diameter > h * 0.75:
                continue


            # -----------------------------------------------
            # score
            #
            # 1. gradient가 강할수록 좋음
            # 2. 서로 반대 방향이어야 함
            # 3. 외곽에 가까운 위치에 약한 가산점
            # -----------------------------------------------

            strength_score = (
                abs(top_gradient_value)
                + abs(bottom_gradient_value)
            )


            outer_score = (
                (upper_end - top_idx)
                + (bottom_idx - lower_start)
            ) * 0.01


            score = (
                strength_score
                + outer_score
            )


            if score > best_score:

                best_score = score

                best_top = int(
                    top_idx
                )

                best_bottom = int(
                    bottom_idx
                )


    if best_top is None:
        return None

    if best_bottom is None:
        return None


    # ========================================================
    # Sub-pixel
    # ========================================================

    top_sub = subpixel_extremum(
        gradient,
        best_top
    )

    bottom_sub = subpixel_extremum(
        gradient,
        best_bottom
    )


    diameter_sub = (
        bottom_sub
        - top_sub
    )


    return (
        top_sub,
        bottom_sub,
        diameter_sub,
        float(gradient[best_top]),
        float(gradient[best_bottom])
    )


# ============================================================
# 8. 여러 X 위치 측정
# ============================================================

x_positions = np.linspace(
    int(roi_w * 0.18),
    int(roi_w * 0.82),
    21
).astype(int)


top_points = []

bottom_points = []

diameters = []

top_signs = []

bottom_signs = []


for x in x_positions:

    result = find_directional_outer_edges(
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

    top_signs.append(
        top_grad
    )

    bottom_signs.append(
        bottom_grad
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
    "========== Directional Edge =========="
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
        "유효 Edge 점이 부족합니다."
    )


# ============================================================
# 9. MAD 이상값 제거
# ============================================================

median_raw = float(
    np.median(
        diameters
    )
)

deviation = np.abs(
    diameters
    - median_raw
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
    "이상값 제거 후 :",
    np.round(
        diameter_valid,
        3
    )
)


if len(diameter_valid) < 7:
    raise RuntimeError(
        "이상값 제거 후 점이 부족합니다."
    )


# ============================================================
# 10. 각각 Line Fitting
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
    abs(top_a - bottom_a)
)


# ============================================================
# 11. 공통 slope
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
# 12. 최종 직선 간 거리
# ============================================================

final_diameter = (
    bottom_intercept
    - top_intercept
)


print()
print(
    "========== Day026-3 최종 =========="
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

print(
    "최종 Line 외경(pixel) :",
    final_diameter
)


# 기준값은 분석용 출력만 사용
REFERENCE_PIXEL = 462.72


print()
print(
    "========== 기존 기준과 비교 =========="
)

print(
    "기준값(pixel) :",
    REFERENCE_PIXEL
)

print(
    "측정값(pixel) :",
    final_diameter
)

print(
    "차이(pixel) :",
    final_diameter
    - REFERENCE_PIXEL
)


# ============================================================
# 13. 시각화
# ============================================================

debug = roi.copy()


# 유효 상단점
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


# 유효 하단점
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


# fitted upper edge
cv2.line(
    debug,
    (x_start, top_start),
    (x_end, top_end),
    (0, 255, 0),
    3
)


# fitted lower edge
cv2.line(
    debug,
    (x_start, bottom_start),
    (x_end, bottom_end),
    (0, 255, 0),
    3
)


# 중앙 외경선
cx = roi_w // 2

top_center = int(
    round(
        common_slope
        * cx
        + top_intercept
    )
)

bottom_center = int(
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
        top_center
    ),
    (
        cx,
        bottom_center
    ),
    (0, 255, 255),
    3
)


cv2.putText(
    debug,
    f"{final_diameter:.3f} px",
    (5, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 255),
    2
)


output_path = (
    OUTPUT_DIR
    / "directional_subpixel_edge.jpg"
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
    "Day026-3 완료"
)

print(
    "=============================="
)