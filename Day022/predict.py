from ultralytics import YOLO
from pathlib import Path


# 1. 학습된 모델 경로
model_path = Path(
    "runs/detect/Day022/runs/measurement_roi_v2/weights/best.pt"
)

model = YOLO(str(model_path))


# 2. 검증 이미지
image_path = Path(
    "Day022/images/train/sample_03.jpg"

)


# 3. YOLO 자동 ROI 검출
results = model.predict(
    source=str(image_path),
    conf=0.001,
    save=True,

    project="Day022/predict_runs",
    name="roi_detection"
)


print()
print("==============================")
print("YOLO 측정 ROI 자동 검출 완료")
print("==============================")


# 4. 검출 결과 좌표 출력
for result in results:

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:

        print("measurement_roi 검출 실패")

        continue


    for box in boxes:

        xyxy = box.xyxy[0].tolist()

        confidence = float(
            box.conf[0]
        )

        class_id = int(
            box.cls[0]
        )

        x1, y1, x2, y2 = xyxy

        print()
        print("class_id :", class_id)
        print("confidence :", confidence)

        print("x1 :", x1)
        print("y1 :", y1)
        print("x2 :", x2)
        print("y2 :", y2)