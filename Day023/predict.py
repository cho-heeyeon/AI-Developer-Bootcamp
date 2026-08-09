from ultralytics import YOLO
from pathlib import Path


# ---------------------------------
# 1. 학습된 모델 경로
# ---------------------------------

model_path = Path(
    "runs/detect/Day023/runs/measurement_roi_aug-2/weights/best.pt"
)

model = YOLO(
    str(model_path)
)


# ---------------------------------
# 2. 테스트 이미지
# ---------------------------------

image_path = Path(
    "Day023/original/sample_07.jpg"
)


# ---------------------------------
# 3. YOLO 자동 ROI 검출
# ---------------------------------

results = model.predict(
    source=str(image_path),

    conf=0.25,

    imgsz=640,

    save=True,

    project="Day023/predict_runs",

    name="roi_detection"
)


print()
print("=" * 40)
print("Day023 YOLO ROI 자동 검출 완료")
print("=" * 40)


# ---------------------------------
# 4. 검출 결과 출력
# ---------------------------------

detection_count = 0


for result in results:

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print()
        print("measurement_roi 검출 실패")

        continue


    for box in boxes:

        detection_count += 1

        class_id = int(
            box.cls[0]
        )

        confidence = float(
            box.conf[0]
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )


        print()
        print(
            "검출 번호 :",
            detection_count
        )

        print(
            "class_id :",
            class_id
        )

        print(
            "confidence :",
            confidence
        )

        print(
            "x1 :",
            x1
        )

        print(
            "y1 :",
            y1
        )

        print(
            "x2 :",
            x2
        )

        print(
            "y2 :",
            y2
        )


## ---------------------------------
# 5. 가장 confidence가 높은 ROI 선택
# ---------------------------------

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


print()
print("=" * 40)
print("최종 측정 ROI 선택")
print("=" * 40)


if best_box is not None:

    x1, y1, x2, y2 = (
        best_box.xyxy[0].tolist()
    )

    print(
        "confidence :",
        best_conf
    )

    print(
        "x1 :",
        x1
    )

    print(
        "y1 :",
        y1
    )

    print(
        "x2 :",
        x2
    )

    print(
        "y2 :",
        y2
    )

    print()
    print(
        "최종 측정 ROI 선택 성공"
    )

else:

    print(
        "측정 ROI 선택 실패"
    )