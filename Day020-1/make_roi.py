import cv2
from pathlib import Path

# -----------------------------
# 1. 폴더 설정
# -----------------------------

base_dir = Path("Day020-1")

sample_files = [
    "sample_03.jpg",
    "sample_04.jpg",
    "sample_06.jpg",
    "sample_07.jpg",
    "sample_N01.jpg",
    "sample_N02.jpg",
]


# -----------------------------
# 2. 이미지별 ROI 선택
# -----------------------------

for filename in sample_files:

    image_path = base_dir / filename

    image = cv2.imread(str(image_path))

    if image is None:
        print("이미지를 찾을 수 없습니다 :", image_path)
        continue

    print()
    print("현재 이미지 :", filename)

    # 화면이 너무 크면 축소해서 표시
    scale = 0.4

    display = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale
    )

    # 마우스로 ROI 선택
    roi_box = cv2.selectROI(
        f"Select ROI - {filename}",
        display,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    x, y, w, h = roi_box

    # ROI를 선택하지 않은 경우
    if w == 0 or h == 0:
        print("ROI 선택 취소 :", filename)
        continue

    # 원본 좌표로 복원
    x = int(x / scale)
    y = int(y / scale)
    w = int(w / scale)
    h = int(h / scale)

    # ROI 추출
    roi = image[
        y:y+h,
        x:x+w
    ]

    # ROI 확인용 이미지
    result = image.copy()

    cv2.rectangle(
        result,
        (x, y),
        (x+w, y+h),
        (0, 0, 255),
        5
    )

    # -----------------------------
    # 3. 파일명 만들기
    # -----------------------------

    stem = image_path.stem

    roi_path = base_dir / f"{stem}_roi.jpg"

    roi_box_path = base_dir / f"{stem}_roi_box.jpg"


    # -----------------------------
    # 4. 저장
    # -----------------------------

    cv2.imwrite(
        str(roi_path),
        roi
    )

    cv2.imwrite(
        str(roi_box_path),
        result
    )


    print("ROI 좌표")
    print("x :", x)
    print("y :", y)
    print("w :", w)
    print("h :", h)

    print("ROI 저장 :", roi_path)
    print("ROI BOX 저장 :", roi_box_path)


print()
print("모든 ROI 작업 완료")