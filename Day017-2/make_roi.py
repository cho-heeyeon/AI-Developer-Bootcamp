import cv2
from pathlib import Path

image_path = Path("Day017-2/sample.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError("sample.jpg를 찾을 수 없습니다.")

scale = 0.4

display = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale
)

roi_box = cv2.selectROI(
    "Select Test ROI",
    display,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x, y, w, h = roi_box

x = int(x / scale)
y = int(y / scale)
w = int(w / scale)
h = int(h / scale)

roi = image[y:y+h, x:x+w]

result = image.copy()

cv2.rectangle(
    result,
    (x, y),
    (x+w, y+h),
    (0, 0, 255),
    5
)

cv2.imwrite(
    "Day017-2/test_roi.jpg",
    roi
)

cv2.imwrite(
    "Day017-2/test_roi_box.jpg",
    result
)

print("x :", x)
print("y :", y)
print("w :", w)
print("h :", h)

print("검증용 ROI 생성 완료")