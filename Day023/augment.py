import cv2
from pathlib import Path


# --------------------------------------------------
# 1. 경로 설정
# --------------------------------------------------

base_dir = Path("Day023")

original_dir = base_dir / "original"

train_image_dir = base_dir / "images" / "train"
val_image_dir = base_dir / "images" / "val"

train_label_dir = base_dir / "labels" / "train"
val_label_dir = base_dir / "labels" / "val"

train_image_dir.mkdir(parents=True, exist_ok=True)
val_image_dir.mkdir(parents=True, exist_ok=True)

train_label_dir.mkdir(parents=True, exist_ok=True)
val_label_dir.mkdir(parents=True, exist_ok=True)


# Day022에서 만든 기존 라벨 사용
day022_train_label_dir = Path("Day022/labels/train")
day022_val_label_dir = Path("Day022/labels/val")


# --------------------------------------------------
# 2. Train / Validation 샘플
# --------------------------------------------------

train_samples = [
    "sample_03",
    "sample_04",
    "sample_06",
    "sample_N01",
    "sample_N02",
]

val_samples = [
    "sample_07"
]


# --------------------------------------------------
# 3. YOLO 라벨 읽기
# --------------------------------------------------

def read_yolo_label(label_path):

    boxes = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            boxes.append([
                class_id,
                x_center,
                y_center,
                width,
                height
            ])

    return boxes


# --------------------------------------------------
# 4. YOLO 라벨 저장
# --------------------------------------------------

def save_yolo_label(label_path, boxes):

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        for box in boxes:

            class_id, xc, yc, w, h = box

            f.write(
                f"{class_id} "
                f"{xc:.6f} "
                f"{yc:.6f} "
                f"{w:.6f} "
                f"{h:.6f}\n"
            )


# --------------------------------------------------
# 5. 한 샘플 증강
# --------------------------------------------------

def augment_sample(
    sample_name,
    label_path,
    output_image_dir,
    output_label_dir
):

    image_path = (
        original_dir
        / f"{sample_name}.jpg"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            "이미지를 찾을 수 없습니다 :",
            image_path
        )

        return


    if not label_path.exists():

        print(
            "라벨을 찾을 수 없습니다 :",
            label_path
        )

        return


    boxes = read_yolo_label(
        label_path
    )


    # --------------------------------------
    # ① 원본
    # --------------------------------------

    cv2.imwrite(
        str(
            output_image_dir
            / f"{sample_name}_orig.jpg"
        ),
        image
    )

    save_yolo_label(
        output_label_dir
        / f"{sample_name}_orig.txt",
        boxes
    )


    # --------------------------------------
    # ② 밝게
    # --------------------------------------

    bright = cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=25
    )

    cv2.imwrite(
        str(
            output_image_dir
            / f"{sample_name}_bright.jpg"
        ),
        bright
    )

    save_yolo_label(
        output_label_dir
        / f"{sample_name}_bright.txt",
        boxes
    )


    # --------------------------------------
    # ③ 어둡게
    # --------------------------------------

    dark = cv2.convertScaleAbs(
        image,
        alpha=1.0,
        beta=-25
    )

    cv2.imwrite(
        str(
            output_image_dir
            / f"{sample_name}_dark.jpg"
        ),
        dark
    )

    save_yolo_label(
        output_label_dir
        / f"{sample_name}_dark.txt",
        boxes
    )


    # --------------------------------------
    # ④ Blur
    # --------------------------------------

    blur = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    cv2.imwrite(
        str(
            output_image_dir
            / f"{sample_name}_blur.jpg"
        ),
        blur
    )

    save_yolo_label(
        output_label_dir
        / f"{sample_name}_blur.txt",
        boxes
    )


    # --------------------------------------
    # ⑤ Contrast
    # --------------------------------------

    contrast = cv2.convertScaleAbs(
        image,
        alpha=1.15,
        beta=0
    )

    cv2.imwrite(
        str(
            output_image_dir
            / f"{sample_name}_contrast.jpg"
        ),
        contrast
    )

    save_yolo_label(
        output_label_dir
        / f"{sample_name}_contrast.txt",
        boxes
    )


    print(
        sample_name,
        "→ 5개 증강 완료"
    )


# --------------------------------------------------
# 6. Train 데이터 생성
# --------------------------------------------------

print()
print("========== TRAIN 증강 시작 ==========")

for sample in train_samples:

    label_path = (
        day022_train_label_dir
        / f"{sample}.txt"
    )

    augment_sample(
        sample,
        label_path,
        train_image_dir,
        train_label_dir
    )


# --------------------------------------------------
# 7. Validation 데이터 생성
# --------------------------------------------------

print()
print("========== VALIDATION 증강 시작 ==========")

for sample in val_samples:

    label_path = (
        day022_val_label_dir
        / f"{sample}.txt"
    )

    augment_sample(
        sample,
        label_path,
        val_image_dir,
        val_label_dir
    )


# --------------------------------------------------
# 8. 생성 개수 확인
# --------------------------------------------------

train_images = list(
    train_image_dir.glob("*.jpg")
)

train_labels = list(
    train_label_dir.glob("*.txt")
)

val_images = list(
    val_image_dir.glob("*.jpg")
)

val_labels = list(
    val_label_dir.glob("*.txt")
)


print()
print("========== 증강 결과 ==========")

print(
    "Train 이미지 :",
    len(train_images)
)

print(
    "Train 라벨 :",
    len(train_labels)
)

print(
    "Val 이미지 :",
    len(val_images)
)

print(
    "Val 라벨 :",
    len(val_labels)
)

print(
    "전체 이미지 :",
    len(train_images)
    + len(val_images)
)

print()
print("==============================")
print("Day023 YOLO 데이터 증강 완료")
print("==============================")