import cv2
from pathlib import Path

# 이미지 읽기
image_path = Path("Day013/shaft.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("이미지를 찾을 수 없습니다.")

# 화면이 너무 크므로 선택용 이미지만 축소
scale = 0.4

display = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale
)

# 마우스로 ROI 선택
roi_box = cv2.selectROI(
    "Select Shaft ROI",
    display,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

# 선택 좌표
x, y, w, h = roi_box

# 원본 이미지 좌표로 복원
x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

print("x :", x)
print("y :", y)
print("w :", w)
print("h :", h)

# ROI 추출
roi = image[
    y:y+h,
    x:x+w
]

# 확인용 이미지
result = image.copy()

cv2.rectangle(
    result,
    (x, y),
    (x+w, y+h),
    (0, 0, 255),
    5
)

# 저장
cv2.imwrite(
    "Day013/shaft_roi_box.jpg",
    result
)

cv2.imwrite(
    "Day013/shaft_roi.jpg",
    roi
)

print("ROI 저장 완료")