from ultralytics import YOLO


# 1. 사전학습 모델
model = YOLO("yolo26n.pt")


# 2. 증강 데이터로 재학습
results = model.train(
    data="Day023/dataset.yaml",

    epochs=100,
    imgsz=640,
    batch=4,

    patience=100,

    project="Day023/runs",
    name="measurement_roi_aug"
)


print()
print("=" * 40)
print("Day023 YOLO 증강 데이터 재학습 완료")
print("=" * 40)