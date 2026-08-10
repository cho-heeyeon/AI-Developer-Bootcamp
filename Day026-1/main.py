import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path("Day026-1")

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

    boxes = result.boxes

    if boxes is None:
        continue

    for box in boxes:

        confidence = float(
            box.conf[0]
        )

        if confidence > best_conf:

            best_conf = confidence
            best_box = box


if best_box is None:

    raise RuntimeError(
        "YOLO measurement ROI 검출 실패"
    )


print()
print("========== YOLO ROI ==========")
print("confidence :", best_conf)


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
        f"이미지를 찾을 수 없습니다 : {IMAGE_PATH}"
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


print("x1 :", x1)
print("y1 :", y1)
print("x2 :", x2)
print("y2 :", y2)


# ============================================================
# 4. Day025 방식의 측정 ROI 확장
# ============================================================

box_height = y2 - y1


vertical_margin = int(
    box_height * 0.40
)


measure_y1 = max(
    0,
    y1 - vertical_margin
)

measure_y2 = min(
    image_h,
    y2 + vertical_margin
)


measure_x1 = x1
measure_x2 = x2


roi = image[
    measure_y1:measure_y2,
    measure_x1:measure_x2
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


gray = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


roi_h, roi_w = gray.shape


# ============================================================
# 6. Sub-pixel 보간
# ============================================================

def subpixel_peak(
    profile,
    index
):

    if index <= 0:
        return float(index)

    if index >= len(profile) - 1:
        return float(index)


    left = float(
        profile[index - 1]
    )

    center = float(
        profile[index]
    )

    right = float(
        profile[index + 1]
    )


    denominator = (
        left
        - 2.0 * center
        + right
    )


    if abs(denominator) < 1e-12:

        return float(index)


    offset = (
        0.5
        * (left - right)
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
# 7. Day025 외곽 Edge 검출 방식
# ============================================================

def find_outer_edges(
    gray,
    x
):

    column = gray[:, x].astype(
        np.float32
    )


    gradient = np.abs(
        np.gradient(column)
    )


    h = len(column)


    # 중앙 내부 반사 Edge를 최대한 피하기 위해
    # 위/아래 검색영역 분리
    upper_end = int(
        h * 0.45
    )

    lower_start = int(
        h * 0.55
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


    # --------------------------------------------------------
    # Day025 방식
    #
    # 각 영역 최대 gradient의 30% 이상 후보를 찾고
    # 상단에서는 첫 후보
    # 하단에서는 마지막 후보 선택
    # --------------------------------------------------------

    upper_max = np.max(
        upper_gradient
    )

    lower_max = np.max(
        lower_gradient
    )


    if upper_max <= 0:
        return None

    if lower_max <= 0:
        return None


    upper_threshold = (
        upper_max * 0.30
    )

    lower_threshold = (
        lower_max * 0.30
    )


    upper_candidates = np.where(
        upper_gradient
        >= upper_threshold
    )[0]


    lower_candidates = np.where(
        lower_gradient
        >= lower_threshold
    )[0]


    if len(upper_candidates) == 0:
        return None


    if len(lower_candidates) == 0:
        return None


    top_index = int(
        upper_candidates[0]
    )


    bottom_index = int(
        lower_candidates[-1]
        + lower_start
    )


    top_sub = subpixel_peak(
        gradient,
        top_index
    )


    bottom_sub = subpixel_peak(
        gradient,
        bottom_index
    )


    diameter = (
        bottom_sub
        - top_sub
    )


    if diameter <= 0:

        return None


    return (
        top_sub,
        bottom_sub,
        diameter
    )


# ============================================================
# 8. Day025보다 많은 위치에서 측정
# ============================================================

x_positions = np.linspace(
    int(roi_w * 0.18),
    int(roi_w * 0.82),
    21
).astype(int)


top_points = []

bottom_points = []

raw_diameters = []


for x in x_positions:

    result = find_outer_edges(
        gray,
        x
    )


    if result is None:

        continue


    top_y, bottom_y, diameter = (
        result
    )


    top_points.append(
        [x, top_y]
    )


    bottom_points.append(
        [x, bottom_y]
    )


    raw_diameters.append(
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

raw_diameters = np.array(
    raw_diameters,
    dtype=np.float64
)


print()
print(
    "========== Day025 방식 Raw 측정 =========="
)

print(
    "유효 점 개수 :",
    len(raw_diameters)
)

print(
    "Raw 측정값(pixel) :",
    np.round(
        raw_diameters,
        3
    )
)


if len(raw_diameters) < 7:

    raise RuntimeError(
        "Line Fitting에 필요한 외곽점이 부족합니다."
    )


# ============================================================
# 9. MAD 기반 이상값 제거
# ============================================================

raw_median = float(
    np.median(
        raw_diameters
    )
)


deviation = np.abs(
    raw_diameters
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
        len(raw_diameters),
        dtype=bool
    )


top_valid = top_points[
    valid_mask
]

bottom_valid = bottom_points[
    valid_mask
]

diameter_valid = raw_diameters[
    valid_mask
]


print()
print(
    "========== 이상값 제거 =========="
)

print(
    "제거 전 중앙값 :",
    raw_median
)

print(
    "MAD :",
    mad
)

print(
    "유효 측정값 :",
    np.round(
        diameter_valid,
        3
    )
)

print(
    "유효 점 개수 :",
    len(diameter_valid)
)


if len(diameter_valid) < 7:

    raise RuntimeError(
        "이상값 제거 후 유효점이 부족합니다."
    )


# ============================================================
# 10. 상단 / 하단 Robust Line Fitting
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


# ============================================================
# 11. 평행성 검사
# ============================================================

slope_difference = abs(
    top_a
    - bottom_a
)


print(
    "기울기 차이 :",
    slope_difference
)


# 현재은 진단용 기준
# 너무 크게 벌어지면 잘못된 Edge를 피팅한 것으로 판단
MAX_SLOPE_DIFFERENCE = 0.08


if slope_difference > MAX_SLOPE_DIFFERENCE:

    print()
    print(
        "경고 : 상단/하단 Edge가 충분히 평행하지 않습니다."
    )

    print(
        "측정 결과를 신뢰하지 마세요."
    )


# ============================================================
# 12. 공통 기울기 계산
#
# 원통 외곽은 상단/하단이 거의 평행해야 하므로
# 두 slope의 평균을 공통 slope로 사용
# ============================================================

common_slope = (
    top_a
    + bottom_a
) / 2.0


# 각 점을 공통 기울기 기준 intercept로 변환
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


# 중앙값으로 robust intercept 계산
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


print()
print(
    "공통 기울기 :",
    common_slope
)


# ============================================================
# 13. 평행한 두 직선 사이 외경
#
# 같은 x 위치에서 y 차이를 계산하면
# 공통 slope이므로 일정한 값이 됨
# ============================================================

measure_x = np.linspace(
    int(roi_w * 0.20),
    int(roi_w * 0.80),
    15
)


fitted_diameters = []


for x in measure_x:

    top_fit = (
        common_slope
        * x
        + top_intercept
    )


    bottom_fit = (
        common_slope
        * x
        + bottom_intercept
    )


    fitted_diameters.append(
        bottom_fit
        - top_fit
    )


fitted_diameters = np.array(
    fitted_diameters,
    dtype=np.float64
)


final_mean = float(
    np.mean(
        fitted_diameters
    )
)


final_median = float(
    np.median(
        fitted_diameters
    )
)


final_std = float(
    np.std(
        fitted_diameters
    )
)


print()
print(
    "========== 최종 외경 =========="
)

print(
    "Line fitting 측정값 :",
    np.round(
        fitted_diameters,
        3
    )
)

print(
    "평균(pixel) :",
    final_mean
)

print(
    "중앙값(pixel) :",
    final_median
)

print(
    "표준편차(pixel) :",
    final_std
)


# ============================================================
# 14. Raw 측정 반복성도 함께 출력
# ============================================================

print()
print(
    "========== Raw 측정 통계 =========="
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


# ============================================================
# 15. 시각화
# ============================================================

debug = roi.copy()


# 실제 상단점
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


# 실제 하단점
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


# 평행 상단 line
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
    (
        x_start,
        top_start
    ),
    (
        x_end,
        top_end
    ),
    (0, 255, 0),
    3
)


cv2.line(
    debug,
    (
        x_start,
        bottom_start
    ),
    (
        x_end,
        bottom_end
    ),
    (0, 255, 0),
    3
)


# 중앙 최종 측정선
center_x = roi_w // 2


center_top = int(
    round(
        common_slope
        * center_x
        + top_intercept
    )
)


center_bottom = int(
    round(
        common_slope
        * center_x
        + bottom_intercept
    )
)


cv2.line(
    debug,
    (
        center_x,
        center_top
    ),
    (
        center_x,
        center_bottom
    ),
    (0, 255, 255),
    3
)


# 결과 표시
cv2.putText(
    debug,
    f"Diameter: {final_median:.3f} px",
    (10, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2
)


output_path = (
    OUTPUT_DIR
    / "corrected_outer_edge_v2.jpg"
)


cv2.imwrite(
    str(output_path),
    debug
)


# ============================================================
# 16. 원본 + 측정 ROI 저장
# ============================================================

overview = image.copy()


cv2.rectangle(
    overview,
    (x1, y1),
    (x2, y2),
    (0, 0, 255),
    4
)


cv2.rectangle(
    overview,
    (
        measure_x1,
        measure_y1
    ),
    (
        measure_x2,
        measure_y2
    ),
    (0, 255, 0),
    3
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "roi_overview.jpg"
    ),
    overview
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
    "Day026-1 보정 완료"
)

print(
    "=============================="
)