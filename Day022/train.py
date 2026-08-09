from ultralytics import YOLO

model = YOLO("yolo26n.pt")

results = model.train(
    data="Day022/dataset.yaml",

    epochs=100,
    imgsz=640,
    batch=2,

    # 작은 데이터셋 진단용
    patience=100,
    close_mosaic=0,

    project="Day022/runs",
    name="measurement_roi_v2"
)

print()
print("YOLO 재학습 완료")