import cv2


def load_image(file_path):
    """이미지 파일을 읽어 반환합니다."""
    image = cv2.imread(file_path)

    if image is None:
        print(f"이미지를 불러올 수 없습니다: {file_path}")
        return None

    return image


def convert_to_gray(image):
    """컬러 이미지를 흑백 이미지로 변환합니다."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def save_image(file_path, image):
    """처리한 이미지를 파일로 저장합니다."""
    success = cv2.imwrite(file_path, image)

    if success:
        print(f"이미지 저장 완료: {file_path}")
    else:
        print(f"이미지 저장 실패: {file_path}")


def main():
    input_path = "Day005/sample.jpg"
    output_path = "Day005/gray_sample.jpg"

    image = load_image(input_path)

    if image is None:
        return

    gray_image = convert_to_gray(image)

    print("원본 이미지 크기:", image.shape)
    print("흑백 이미지 크기:", gray_image.shape)

    save_image(output_path, gray_image)

    cv2.imshow("Original Image", image)
    cv2.imshow("Gray Image", gray_image)

    print("이미지 창에서 아무 키나 누르면 종료됩니다.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()