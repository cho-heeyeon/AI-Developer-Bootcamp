import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# -----------------------------------------
# 1. 경로 설정
# -----------------------------------------

base_dir = Path("Day024")

image_path = base_dir / "sample_07.jpg"

output_dir = base_dir / "output"
output_dir.mkdir(exist_ok=True)


# Day023에서 학습한 YOLO 모델
model_path = Path(
    "runs/detect/Day023/runs/"
    "measurement_roi_aug-2/weights/best.pt"
)


# -----------------------------------------
# 2. YOLO 모델 불러오기
# -----------------------------------------

model = YOLO(
    str(model_path)
)


# -----------------------------------------
# 3. 원본 이미지 읽기
# -----------------------------------------

image = cv2.imread(
    str(image_path)
)

if image is None:

    raise FileNotFoundError(
        f"이미지를 찾을 수 없습니다 : {image_path}"
    )


# -----------------------------------------
# 4. YOLO ROI 검출
# -----------------------------------------

results = model.predict(
    source=str(image_path),
    conf=0.25,
    imgsz=640,
    verbose=False
)


# -----------------------------------------
# 5. 가장 confidence 높은 ROI 선택
# -----------------------------------------

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
        "measurement_roi를 검출하지 못했습니다."
    )


# -----------------------------------------
# 6. Bounding Box 좌표
# -----------------------------------------

x1, y1, x2, y2 = (
    best_box.xyxy[0]
    .cpu()
    .numpy()
    .astype(int)
)


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


# -----------------------------------------
# 7. YOLO ROI 자동 Crop
# -----------------------------------------

roi = image[
    y1:y2,
    x1:x2
]


if roi.size == 0:

    raise RuntimeError(
        "ROI Crop 결과가 비어 있습니다."
    )


roi_path = (
    output_dir
    / "yolo_roi.jpg"
)

cv2.imwrite(
    str(roi_path),
    roi
)


# -----------------------------------------
# 8. ROI 확인용 Bounding Box
# -----------------------------------------

box_result = image.copy()

cv2.rectangle(
    box_result,
    (x1, y1),
    (x2, y2),
    (0, 0, 255),
    5
)

cv2.putText(
    box_result,
    f"ROI {best_conf:.3f}",
    (x1, max(30, y1 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 255),
    3
)

cv2.imwrite(
    str(
        output_dir
        / "yolo_detection.jpg"
    ),
    box_result
)


# -----------------------------------------
# 9. Sub-pixel 계산 함수
# -----------------------------------------

def subpixel_peak(
    profile,
    index
):

    if (
        index <= 0
        or index >= len(profile) - 1
    ):

        return float(index)


    y_prev = profile[index - 1]
    y_center = profile[index]
    y_next = profile[index + 1]


    denominator = (
        y_prev
        - 2 * y_center
        + y_next
    )


    if denominator == 0:

        return float(index)


    offset = (
        0.5
        * (y_prev - y_next)
        / denominator
    )


    return float(
        index + offset
    )


# -----------------------------------------
# 10. ROI 전처리
# -----------------------------------------

gray = cv2.cvtColor(
    roi,
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


# -----------------------------------------
# 11. 중앙부 7개 위치 측정
# -----------------------------------------

x_start = int(
    width * 0.25
)

x_end = int(
    width * 0.75
)

x_positions = np.linspace(
    x_start,
    x_end,
    7,
    dtype=int
)


diameters = []

measure_result = roi.copy()


for x in x_positions:

    # 주변 5 pixel 평균
    strip_start = max(
        0,
        x - 2
    )

    strip_end = min(
        width,
        x + 3
    )


    strip = gradient_abs[
        :,
        strip_start:strip_end
    ]


    profile = np.mean(
        strip,
        axis=1
    )


    # ---------------------------------
    # 상단 Edge
    # ---------------------------------

    top_profile = profile[
        :center_y
    ]

    top_index = int(
        np.argmax(
            top_profile
        )
    )

    y_top = subpixel_peak(
        top_profile,
        top_index
    )


    # ---------------------------------
    # 하단 Edge
    # ---------------------------------

    bottom_profile = profile[
        center_y:
    ]

    bottom_local = int(
        np.argmax(
            bottom_profile
        )
    )

    bottom_index = (
        center_y
        + bottom_local
    )

    y_bottom = subpixel_peak(
        profile,
        bottom_index
    )


    # ---------------------------------
    # 외경 계산
    # ---------------------------------

    diameter = (
        y_bottom
        - y_top
    )

    diameters.append(
        diameter
    )


    # ---------------------------------
    # 측정선 표시
    # ---------------------------------

    y_top_draw = int(
        round(y_top)
    )

    y_bottom_draw = int(
        round(y_bottom)
    )


    # 상단 빨강
    cv2.circle(
        measure_result,
        (x, y_top_draw),
        4,
        (0, 0, 255),
        -1
    )


    # 하단 파랑
    cv2.circle(
        measure_result,
        (x, y_bottom_draw),
        4,
        (255, 0, 0),
        -1
    )


    # 외경 측정선
    cv2.line(
        measure_result,
        (x, y_top_draw),
        (x, y_bottom_draw),
        (0, 255, 0),
        2
    )


# -----------------------------------------
# 12. 대표 외경
# -----------------------------------------

diameters = np.array(
    diameters
)

median_px = float(
    np.median(
        diameters
    )
)

mean_px = float(
    np.mean(
        diameters
    )
)

std_px = float(
    np.std(
        diameters
    )
)


print()
print(
    "========== Sub-pixel 외경 측정 =========="
)

print(
    "측정값 :",
    diameters
)

print(
    "외경 중앙값(pixel) :",
    median_px
)

print(
    "외경 평균(pixel) :",
    mean_px
)

print(
    "외경 표준편차(pixel) :",
    std_px
)


# -----------------------------------------
# 13. 결과 저장
# -----------------------------------------

measurement_path = (
    output_dir
    / "subpixel_measurement.jpg"
)

cv2.imwrite(
    str(measurement_path),
    measure_result
)


print()
print(
    "YOLO ROI 저장 :",
    roi_path
)

print(
    "Sub-pixel 결과 저장 :",
    measurement_path
)

print()
print("==============================")
print("Day024 자동 측정 완료")
print("==============================")