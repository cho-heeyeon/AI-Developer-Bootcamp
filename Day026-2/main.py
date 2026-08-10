import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로
# ============================================================

BASE_DIR = Path("Day026-2")

IMAGE_PATH = BASE_DIR / "sample_07.jpg"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path(
    "runs/detect/Day023/runs/"
    "measurement_roi_aug-2/weights/best.pt"
)


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

image = cv2.imread(str(IMAGE_PATH))

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


# ============================================================
# 4. Day025 방식 ROI 확장
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


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "analysis_roi.jpg"
    ),
    roi
)


# ============================================================
# 5. Gray / Blur
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
# 6. 중앙 X 위치 선정
# ============================================================

center_x = roi_w // 2


# 주변 7 pixel 평균 사용
strip = blur[
    :,
    max(0, center_x - 3):
    min(roi_w, center_x + 4)
]


profile = np.mean(
    strip,
    axis=1
)


# ============================================================
# 7. Gradient 계산
# ============================================================

gradient_signed = np.gradient(
    profile
)

gradient_abs = np.abs(
    gradient_signed
)


# ============================================================
# 8. 상단 / 하단 후보 peak 출력
# ============================================================

center_y = roi_h // 2


top_gradient = gradient_abs[
    :center_y
]

bottom_gradient = gradient_abs[
    center_y:
]


# 가장 강한 후보 10개씩
top_candidates = np.argsort(
    top_gradient
)[-10:][::-1]


bottom_candidates_local = np.argsort(
    bottom_gradient
)[-10:][::-1]


bottom_candidates = (
    bottom_candidates_local
    + center_y
)


print()
print(
    "========== 상단 Edge 후보 =========="
)

for idx in top_candidates:

    print(
        f"y={idx:4d}",
        f"gradient={gradient_abs[idx]:.3f}",
        f"brightness={profile[idx]:.3f}"
    )


print()
print(
    "========== 하단 Edge 후보 =========="
)

for idx in bottom_candidates:

    print(
        f"y={idx:4d}",
        f"gradient={gradient_abs[idx]:.3f}",
        f"brightness={profile[idx]:.3f}"
    )


# ============================================================
# 9. Day026-1 방식 외곽 Edge 재현
# ============================================================

upper_end = int(
    roi_h * 0.45
)

lower_start = int(
    roi_h * 0.55
)


upper_gradient = gradient_abs[
    :upper_end
]

lower_gradient = gradient_abs[
    lower_start:
]


upper_threshold = (
    np.max(upper_gradient)
    * 0.30
)

lower_threshold = (
    np.max(lower_gradient)
    * 0.30
)


upper_candidates_threshold = np.where(
    upper_gradient
    >= upper_threshold
)[0]


lower_candidates_threshold = np.where(
    lower_gradient
    >= lower_threshold
)[0]


top_selected = int(
    upper_candidates_threshold[0]
)


bottom_selected = int(
    lower_candidates_threshold[-1]
    + lower_start
)


diameter_selected = (
    bottom_selected
    - top_selected
)


print()
print(
    "========== 현재 알고리즘 선택 =========="
)

print(
    "상단 선택 y :",
    top_selected
)

print(
    "하단 선택 y :",
    bottom_selected
)

print(
    "현재 외경(pixel) :",
    diameter_selected
)


# ============================================================
# 10. 기준값과 차이
# ============================================================

REFERENCE_PIXEL = 462.72


difference = (
    diameter_selected
    - REFERENCE_PIXEL
)


print()
print(
    "========== 기준 비교 =========="
)

print(
    "기준 외경(pixel) :",
    REFERENCE_PIXEL
)

print(
    "현재 외경(pixel) :",
    diameter_selected
)

print(
    "차이(pixel) :",
    difference
)


# ============================================================
# 11. 분석 이미지
# ============================================================

debug = roi.copy()


# 중앙 분석선
cv2.line(
    debug,
    (center_x, 0),
    (center_x, roi_h - 1),
    (0, 255, 255),
    2
)


# 현재 선택 상단
cv2.line(
    debug,
    (0, top_selected),
    (roi_w - 1, top_selected),
    (0, 0, 255),
    2
)


# 현재 선택 하단
cv2.line(
    debug,
    (0, bottom_selected),
    (roi_w - 1, bottom_selected),
    (255, 0, 0),
    2
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "selected_edges.jpg"
    ),
    debug
)


# ============================================================
# 12. 밝기 프로파일 이미지 생성
# ============================================================

profile_img = np.zeros(
    (roi_h, 600, 3),
    dtype=np.uint8
)


# 밝기 normalize
profile_norm = cv2.normalize(
    profile,
    None,
    0,
    250,
    cv2.NORM_MINMAX
)


gradient_norm = cv2.normalize(
    gradient_abs,
    None,
    0,
    250,
    cv2.NORM_MINMAX
)

profile_norm = profile_norm.flatten()
gradient_norm = gradient_norm.flatten()

for y in range(
    roi_h - 1
):

    # 밝기 profile
    x_p1 = int(
        profile_norm[y]
    )

    x_p2 = int(
        profile_norm[y + 1]
    )


    cv2.line(
        profile_img,
        (x_p1, y),
        (x_p2, y + 1),
        (255, 255, 255),
        1
    )


    # gradient
    x_g1 = 300 + int(
        gradient_norm[y]
    )

    x_g2 = 300 + int(
        gradient_norm[y + 1]
    )


    cv2.line(
        profile_img,
        (x_g1, y),
        (x_g2, y + 1),
        (0, 255, 0),
        1
    )


# 선택 Edge 표시
cv2.line(
    profile_img,
    (0, top_selected),
    (599, top_selected),
    (0, 0, 255),
    2
)


cv2.line(
    profile_img,
    (0, bottom_selected),
    (599, bottom_selected),
    (255, 0, 0),
    2
)


cv2.imwrite(
    str(
        OUTPUT_DIR
        / "edge_profile.jpg"
    ),
    profile_img
)


print()
print(
    "분석 이미지 저장 완료"
)

print(
    OUTPUT_DIR
    / "selected_edges.jpg"
)

print(
    OUTPUT_DIR
    / "edge_profile.jpg"
)

print()
print("==============================")
print("Day026-2 원인 분석 완료")
print("==============================")