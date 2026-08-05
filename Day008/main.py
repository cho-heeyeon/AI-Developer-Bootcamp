import cv2
from pathlib import Path


def load_image(file_path):
    """이미지를 불러옵니다."""
    image = cv2.imread(str(file_path))

    if image is None:
        print(f"이미지를 읽을 수 없습니다: {file_path}")
        return None

    return image


def detect_edges(image, low_threshold=50, high_threshold=150):
    """이미지를 흑백으로 변환한 뒤 경계선을 검출합니다."""

    # 1. 컬러 이미지를 흑백 이미지로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 작은 잡음을 줄이기 위해 흐림 처리
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # 3. Canny 알고리즘으로 경계선 검출
    edge_image = cv2.Canny(
        blurred_image,
        low_threshold,
        high_threshold
    )

    return gray_image, blurred_image, edge_image


def main():
    image_path = Path("Day008/sample.jpg")
    output_path = Path("Day008/edge_sample.jpg")

    image = load_image(image_path)

    if image is None:
        return

    gray_image, blurred_image, edge_image = detect_edges(image)

    success = cv2.imwrite(str(output_path), edge_image)

    print("-" * 40)
    print("Canny Edge Detection 실행 결과")
    print("-" * 40)
    print(f"원본 이미지 크기: {image.shape}")
    print(f"흑백 이미지 크기: {gray_image.shape}")
    print("낮은 임계값: 50")
    print("높은 임계값: 150")

    if success:
        print(f"에지 이미지 저장 완료: {output_path}")
    else:
        print("에지 이미지 저장에 실패했습니다.")

    cv2.imshow("Original Image", image)
    cv2.imshow("Gray Image", gray_image)
    cv2.imshow("Blurred Image", blurred_image)
    cv2.imshow("Edge Image", edge_image)

    print("-" * 40)
    print("아무 키나 누르면 종료됩니다.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()