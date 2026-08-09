import cv2
from pathlib import Path


# ---------------------------------
# 1. 경로 설정
# ---------------------------------

base_dir = Path("Day023")

train_image_dir = base_dir / "images" / "train"
train_label_dir = base_dir / "labels" / "train"

val_image_dir = base_dir / "images" / "val"
val_label_dir = base_dir / "labels" / "val"

output_dir = base_dir / "verified"

output_dir.mkdir(exist_ok=True)


# ---------------------------------
# 2. 검증할 대표 이미지
# ---------------------------------

samples = [
    (
        "sample_03_rot2",
        train_image_dir,
        train_label_dir
    ),

    (
        "sample_N01_rot2",
        train_image_dir,
        train_label_dir
    ),

    (
        "sample_07_rot2",
        val_image_dir,
        val_label_dir
    ),
]


# ---------------------------------
# 3. 이미지 / 라벨 검증
# ---------------------------------

for sample_name, image_dir, label_dir in samples:

    image_path = (
        image_dir
        / f"{sample_name}.jpg"
    )

    label_path = (
        label_dir
        / f"{sample_name}.txt"
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            "이미지를 찾을 수 없습니다 :",
            image_path
        )

        continue


    if not label_path.exists():

        print(
            "라벨을 찾을 수 없습니다 :",
            label_path
        )

        continue


    height, width = image.shape[:2]

    result = image.copy()


    # ---------------------------------
    # 4. YOLO 라벨 읽기
    # ---------------------------------

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()


    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue


        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])

        box_width = float(parts[3])
        box_height = float(parts[4])


        # ---------------------------------
        # 5. 정규화 좌표 → pixel
        # ---------------------------------

        center_x_px = (
            x_center * width
        )

        center_y_px = (
            y_center * height
        )

        width_px = (
            box_width * width
        )

        height_px = (
            box_height * height
        )


        x1 = int(
            center_x_px
            - width_px / 2
        )

        y1 = int(
            center_y_px
            - height_px / 2
        )

        x2 = int(
            center_x_px
            + width_px / 2
        )

        y2 = int(
            center_y_px
            + height_px / 2
        )


        # ---------------------------------
        # 6. Bounding Box 표시
        # ---------------------------------

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            5
        )


        cv2.putText(
            result,
            "measurement_roi",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


        print()
        print("Sample :", sample_name)

        print(
            "class_id :",
            class_id
        )

        print(
            "box :",
            x1,
            y1,
            x2,
            y2
        )


    # ---------------------------------
    # 7. 저장
    # ---------------------------------

    output_path = (
        output_dir
        / f"{sample_name}_verified.jpg"
    )

    cv2.imwrite(
        str(output_path),
        result
    )

    print(
        "검증 이미지 저장 :",
        output_path
    )


print()
print("==============================")
print("증강 라벨 검증 완료")
print("==============================")