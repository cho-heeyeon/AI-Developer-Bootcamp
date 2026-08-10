import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로
# ============================================================

BASE_DIR = Path("Day026-6")
IMAGE_PATH = BASE_DIR / "sample_07.jpg"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(
    "runs/detect/Day023/runs/"
    "measurement_roi_aug-2/weights/best.pt"
)

REFERENCE_PIXEL = 462.72


# ============================================================
# 2. YOLO ROI
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
    raise RuntimeError("YOLO ROI 검출 실패")


if best_conf < 0.70:
    raise RuntimeError(
        f"YOLO confidence 부족 : {best_conf:.3f}"
    )


# ============================================================
# 3. 이미지
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)


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


# ============================================================
# 4. Day026-5와 같은 ROI
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
    raise RuntimeError("ROI가 비어 있습니다.")


cv2.imwrite(
    str(OUTPUT_DIR / "analysis_roi.jpg"),
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
# 6. Sub-pixel
# ============================================================

def subpixel_edge(
    abs_gradient,
    index
):

    if index <= 0:
        return float(index)

    if index >= len(abs_gradient) - 1:
        return float(index)

    g1 = float(abs_gradient[index - 1])
    g2 = float(abs_gradient[index])
    g3 = float(abs_gradient[index + 1])

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
# 7. Day026-5 Edge 선택 그대로
# ============================================================

def find_edge_and_profile(
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


            # 방향이 반대
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
        if p[3] >= strength_limit
    ]


    if len(strong_pairs) == 0:
        strong_pairs = candidate_pairs


    best_pair = max(
        strong_pairs,
        key=lambda p: p[2]
    )


    top_idx = best_pair[0]
    bottom_idx = best_pair[1]


    top_sub = subpixel_edge(
        abs_gradient,
        top_idx
    )

    bottom_sub = subpixel_edge(
        abs_gradient,
        bottom_idx
    )


    return {
        "x": x,

        "profile": profile,

        "gradient": gradient,

        "abs_gradient": abs_gradient,

        "top_idx": top_idx,

        "bottom_idx": bottom_idx,

        "top_sub": top_sub,

        "bottom_sub": bottom_sub,

        "diameter": (
            bottom_sub
            - top_sub
        )
    }


# ============================================================
# 8. 분석할 X 위치
#
# 중앙부 9개 위치
# ============================================================

x_positions = np.linspace(
    int(roi_w * 0.25),
    int(roi_w * 0.75),
    9
).astype(int)


measurements = []


for x in x_positions:

    result = find_edge_and_profile(
        blur,
        x
    )

    if result is not None:
        measurements.append(
            result
        )


if len(measurements) < 5:

    raise RuntimeError(
        "분석 가능한 측정점 부족"
    )


# ============================================================
# 9. 측정값 출력
# ============================================================

diameters = np.array(
    [
        m["diameter"]
        for m in measurements
    ],
    dtype=np.float64
)


print()
print(
    "========== Day026-6 측정 =========="
)


for m in measurements:

    print()

    print(
        "x :",
        m["x"]
    )

    print(
        "top integer :",
        m["top_idx"]
    )

    print(
        "top subpixel :",
        m["top_sub"]
    )

    print(
        "bottom integer :",
        m["bottom_idx"]
    )

    print(
        "bottom subpixel :",
        m["bottom_sub"]
    )

    print(
        "diameter :",
        m["diameter"]
    )


print()
print(
    "========== 측정 통계 =========="
)

print(
    "Median :",
    np.median(diameters)
)

print(
    "Mean :",
    np.mean(diameters)
)

print(
    "STD :",
    np.std(diameters)
)

print(
    "기준값 :",
    REFERENCE_PIXEL
)

print(
    "Median 차이 :",
    np.median(diameters)
    - REFERENCE_PIXEL
)


# ============================================================
# 10. Integer와 Sub-pixel 차이 분석
#
# 실제 1.35 px가
# Sub-pixel 보간 자체 때문인지 확인
# ============================================================

integer_diameters = np.array(
    [
        (
            m["bottom_idx"]
            - m["top_idx"]
        )
        for m in measurements
    ],
    dtype=np.float64
)


subpixel_effects = (
    diameters
    - integer_diameters
)


print()
print(
    "========== Sub-pixel 영향 =========="
)

print(
    "Integer 외경 :",
    integer_diameters
)

print(
    "Sub-pixel 외경 :",
    np.round(
        diameters,
        4
    )
)

print(
    "Sub-pixel 이동량 :",
    np.round(
        subpixel_effects,
        4
    )
)

print(
    "평균 Sub-pixel 영향 :",
    np.mean(
        subpixel_effects
    )
)


# ============================================================
# 11. 상단 / 하단 각각 이동량
# ============================================================

top_offsets = np.array(
    [
        m["top_sub"]
        - m["top_idx"]
        for m in measurements
    ]
)


bottom_offsets = np.array(
    [
        m["bottom_sub"]
        - m["bottom_idx"]
        for m in measurements
    ]
)


print()
print(
    "========== 상/하단 Sub-pixel Offset =========="
)

print(
    "상단 offset :",
    np.round(
        top_offsets,
        4
    )
)

print(
    "하단 offset :",
    np.round(
        bottom_offsets,
        4
    )
)

print(
    "상단 평균 offset :",
    np.mean(
        top_offsets
    )
)

print(
    "하단 평균 offset :",
    np.mean(
        bottom_offsets
    )
)


# ============================================================
# 12. Edge 주변 확대 이미지
# ============================================================

debug = roi.copy()


for m in measurements:

    x = int(
        m["x"]
    )


    top_y = int(
        round(
            m["top_sub"]
        )
    )


    bottom_y = int(
        round(
            m["bottom_sub"]
        )
    )


    # 상단
    cv2.circle(
        debug,
        (x, top_y),
        4,
        (0, 0, 255),
        -1
    )


    # 하단
    cv2.circle(
        debug,
        (x, bottom_y),
        4,
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


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "edge_points_analysis.jpg"
    ),
    debug
)


# ============================================================
# 13. 중앙 측정 위치 Profile 분석
# ============================================================

center_measurement = measurements[
    len(measurements) // 2
]


profile = center_measurement[
    "profile"
]

gradient = center_measurement[
    "gradient"
]

abs_gradient = center_measurement[
    "abs_gradient"
]


top_idx = center_measurement[
    "top_idx"
]

bottom_idx = center_measurement[
    "bottom_idx"
]


# ============================================================
# 14. 선택 Edge 주변 ±8 px 출력
# ============================================================

SEARCH_RADIUS = 8


print()
print(
    "========== 중앙 X 상단 주변 =========="
)


for y in range(
    max(0, top_idx - SEARCH_RADIUS),
    min(
        roi_h,
        top_idx + SEARCH_RADIUS + 1
    )
):

    marker = (
        "<-- selected"
        if y == top_idx
        else ""
    )

    print(
        f"y={y:4d}",
        f"brightness={profile[y]:8.3f}",
        f"grad={gradient[y]:8.3f}",
        f"|grad|={abs_gradient[y]:8.3f}",
        marker
    )


print()
print(
    "========== 중앙 X 하단 주변 =========="
)


for y in range(
    max(
        0,
        bottom_idx - SEARCH_RADIUS
    ),
    min(
        roi_h,
        bottom_idx + SEARCH_RADIUS + 1
    )
):

    marker = (
        "<-- selected"
        if y == bottom_idx
        else ""
    )

    print(
        f"y={y:4d}",
        f"brightness={profile[y]:8.3f}",
        f"grad={gradient[y]:8.3f}",
        f"|grad|={abs_gradient[y]:8.3f}",
        marker
    )


# ============================================================
# 15. Profile 시각화
# ============================================================

PROFILE_WIDTH = 800


profile_img = np.zeros(
    (
        roi_h,
        PROFILE_WIDTH,
        3
    ),
    dtype=np.uint8
)


profile_norm = cv2.normalize(
    profile,
    None,
    0,
    300,
    cv2.NORM_MINMAX
).flatten()


gradient_norm = cv2.normalize(
    abs_gradient,
    None,
    0,
    300,
    cv2.NORM_MINMAX
).flatten()


for y in range(
    roi_h - 1
):

    # 밝기
    px1 = int(
        profile_norm[y]
    )

    px2 = int(
        profile_norm[y + 1]
    )


    cv2.line(
        profile_img,
        (px1, y),
        (px2, y + 1),
        (255, 255, 255),
        1
    )


    # Gradient
    gx1 = (
        420
        + int(
            gradient_norm[y]
        )
    )

    gx2 = (
        420
        + int(
            gradient_norm[y + 1]
        )
    )


    cv2.line(
        profile_img,
        (gx1, y),
        (gx2, y + 1),
        (0, 255, 0),
        1
    )


# integer edge
cv2.line(
    profile_img,
    (0, top_idx),
    (
        PROFILE_WIDTH - 1,
        top_idx
    ),
    (0, 0, 255),
    1
)


cv2.line(
    profile_img,
    (0, bottom_idx),
    (
        PROFILE_WIDTH - 1,
        bottom_idx
    ),
    (255, 0, 0),
    1
)


# Sub-pixel 위치는 이미지상 가까운 정수로 표시
top_sub_draw = int(
    round(
        center_measurement["top_sub"]
    )
)

bottom_sub_draw = int(
    round(
        center_measurement["bottom_sub"]
    )
)


cv2.circle(
    profile_img,
    (
        400,
        top_sub_draw
    ),
    5,
    (0, 255, 255),
    -1
)


cv2.circle(
    profile_img,
    (
        400,
        bottom_sub_draw
    ),
    5,
    (0, 255, 255),
    -1
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "edge_offset_profile.jpg"
    ),
    profile_img
)


# ============================================================
# 16. 종료
# ============================================================

print()
print(
    "분석 이미지 저장 :"
)

print(
    OUTPUT_DIR
    / "edge_points_analysis.jpg"
)

print(
    OUTPUT_DIR
    / "edge_offset_profile.jpg"
)


print()
print(
    "=============================="
)

print(
    "Day026-6 원인 분석 완료"
)

print(
    "=============================="
)