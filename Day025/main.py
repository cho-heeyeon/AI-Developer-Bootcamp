from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path("Day025")

IMAGE_PATH = BASE_DIR / "sample_07.jpg"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Day023에서 학습한 YOLO 모델
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

    print("YOLO ROI 검출 실패")
    raise SystemExit


# ============================================================
# 3. YOLO 신뢰도 검사
# ============================================================

print()
print("========== YOLO ROI ==========")
print("confidence :", best_conf)


if best_conf < 0.70:

    print("ROI 신뢰도 부족")
    print("측정을 중단합니다.")

    raise SystemExit


# ============================================================
# 4. 원본 이미지
# ============================================================

image = cv2.imread(str(IMAGE_PATH))

if image is None:

    print("이미지를 읽을 수 없습니다.")
    raise SystemExit


image_h, image_w = image.shape[:2]


x1, y1, x2, y2 = (
    best_box.xyxy[0]
    .cpu()
    .numpy()
    .astype(int)
)


# 이미지 범위 보호
x1 = max(0, x1)
y1 = max(0, y1)

x2 = min(image_w, x2)
y2 = min(image_h, y2)


print("x1 :", x1)
print("y1 :", y1)
print("x2 :", x2)
print("y2 :", y2)


# ============================================================
# 5. ROI를 상하 방향으로 확장
#
# 중요:
# YOLO 박스 자체를 외경으로 사용하지 않습니다.
#
# 실제 외곽 Edge가 YOLO 박스 밖에 있을 수 있으므로
# 측정용 ROI를 위/아래로 확장합니다.
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


# X 방향은 YOLO가 찾은 위치 사용
measure_x1 = x1
measure_x2 = x2


roi = image[
    measure_y1:measure_y2,
    measure_x1:measure_x2
].copy()


cv2.imwrite(
    str(OUTPUT_DIR / "expanded_roi.jpg"),
    roi
)


# ============================================================
# 6. 전처리
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
# 7. Sub-pixel 함수
#
# 세 점을 이용한 포물선 보간
# ============================================================

def subpixel_peak(values, index):

    if index <= 0:
        return float(index)

    if index >= len(values) - 1:
        return float(index)

    left = float(values[index - 1])
    center = float(values[index])
    right = float(values[index + 1])

    denominator = (
        left
        - 2.0 * center
        + right
    )

    if abs(denominator) < 1e-12:
        return float(index)

    offset = 0.5 * (
        left - right
    ) / denominator

    # 비정상적인 보간 방지
    offset = np.clip(
        offset,
        -1.0,
        1.0
    )

    return float(index) + float(offset)


# ============================================================
# 8. 외곽 Edge 찾기
#
# 기존 방식:
# 가장 강한 Edge 선택
#
# 문제:
# 반사광/가공면 Edge가 더 강하면 잘못 선택
#
# Day025 방식:
# 위쪽 영역에서 실제 외곽 후보 탐색
# 아래쪽 영역에서 실제 외곽 후보 탐색
# ============================================================

def find_outer_edges(gray, x):

    column = gray[:, x].astype(
        np.float32
    )


    # ----------------------------------------
    # 수직 방향 밝기 변화량
    # ----------------------------------------

    gradient = np.abs(
        np.gradient(column)
    )


    h = len(column)


    # ----------------------------------------
    # 위/아래 검색 영역 분리
    #
    # 중앙부의 반사 Edge를 최대한 제외
    # ----------------------------------------

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


    # ----------------------------------------
    # 위쪽 외곽 후보
    # ----------------------------------------

    upper_max = np.max(
        upper_gradient
    )

    upper_threshold = (
        upper_max * 0.30
    )


    upper_candidates = np.where(
        upper_gradient
        >= upper_threshold
    )[0]


    if len(upper_candidates) == 0:
        return None


    # 바깥 → 안쪽
    # 첫 번째 유효 Edge
    top_index = int(
        upper_candidates[0]
    )


    # ----------------------------------------
    # 아래쪽 외곽 후보
    # ----------------------------------------

    lower_max = np.max(
        lower_gradient
    )

    lower_threshold = (
        lower_max * 0.30
    )


    lower_candidates = np.where(
        lower_gradient
        >= lower_threshold
    )[0]


    if len(lower_candidates) == 0:
        return None


    # 아래에서 위로 탐색하므로
    # 마지막 후보 사용
    bottom_index = int(
        lower_candidates[-1]
        + lower_start
    )


    # ----------------------------------------
    # Sub-pixel 보정
    # ----------------------------------------

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
# 9. 여러 X 위치에서 반복 측정
# ============================================================

measurement_count = 7


# ROI 양 끝은 피합니다.
x_positions = np.linspace(
    int(roi_w * 0.20),
    int(roi_w * 0.80),
    measurement_count
).astype(int)


measurements = []

debug_image = roi.copy()


for x in x_positions:

    result = find_outer_edges(
        gray,
        x
    )

    if result is None:

        print(
            "Edge 검출 실패 x =",
            x
        )

        continue


    top_y, bottom_y, diameter = result


    measurements.append(
        diameter
    )


    # ----------------------------------------
    # 결과 표시
    # ----------------------------------------

    cv2.circle(
        debug_image,
        (
            int(x),
            int(round(top_y))
        ),
        4,
        (0, 0, 255),
        -1
    )


    cv2.circle(
        debug_image,
        (
            int(x),
            int(round(bottom_y))
        ),
        4,
        (255, 0, 0),
        -1
    )


    cv2.line(
        debug_image,
        (
            int(x),
            int(round(top_y))
        ),
        (
            int(x),
            int(round(bottom_y))
        ),
        (0, 255, 0),
        2
    )


# ============================================================
# 10. 측정 결과
# ============================================================

measurements = np.array(
    measurements,
    dtype=np.float64
)


print()
print(
    "========== Day025 외곽 측정 =========="
)


if len(measurements) == 0:

    print(
        "유효한 외경 측정값이 없습니다."
    )

    raise SystemExit


median_value = np.median(
    measurements
)

mean_value = np.mean(
    measurements
)

std_value = np.std(
    measurements
)


print(
    "측정값 :",
    np.round(measurements, 3)
)

print(
    "외경 중앙값(pixel) :",
    median_value
)

print(
    "외경 평균(pixel) :",
    mean_value
)

print(
    "외경 표준편차(pixel) :",
    std_value
)


# ============================================================
# 11. 이상값 검사
#
# 중앙값에서 너무 멀리 떨어진 측정값 제거
# ============================================================

tolerance = 5.0


valid_mask = (
    np.abs(
        measurements
        - median_value
    )
    <= tolerance
)


valid_measurements = measurements[
    valid_mask
]


print()
print(
    "========== 이상값 제거 =========="
)

print(
    "유효 측정값 :",
    np.round(
        valid_measurements,
        3
    )
)


if len(valid_measurements) > 0:

    final_median = np.median(
        valid_measurements
    )

    final_std = np.std(
        valid_measurements
    )


    print(
        "최종 외경 중앙값(pixel) :",
        final_median
    )

    print(
        "최종 표준편차(pixel) :",
        final_std
    )


# ============================================================
# 12. 결과 이미지 저장
# ============================================================

cv2.imwrite(
    str(
        OUTPUT_DIR
        / "outer_edge_measurement.jpg"
    ),
    debug_image
)


# 원본에 YOLO ROI 표시
yolo_debug = image.copy()


cv2.rectangle(
    yolo_debug,
    (x1, y1),
    (x2, y2),
    (0, 0, 255),
    4
)


cv2.rectangle(
    yolo_debug,
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
        / "yolo_and_measure_roi.jpg"
    ),
    yolo_debug
)


print()
print(
    "결과 이미지 저장 :",
    OUTPUT_DIR
)

print()
print(
    "=============================="
)

print(
    "Day025 완료"
)

print(
    "=============================="
)