import cv2
from pathlib import Path


def main():

    image_path = Path("Day009/sample.jpg")

    image = cv2.imread(str(image_path))

    if image is None:
        print("이미지를 읽을 수 없습니다.")
        return

    # 흑백 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge
    edge = cv2.Canny(blur, 50, 150)

    # Contour 찾기
    contours, hierarchy = cv2.findContours(
        edge,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = image.copy()

    cv2.drawContours(
        result,
        contours,
        -1,
        (0, 0, 255),
        2
    )

    print("-" * 40)
    print(f"찾은 Contour 개수 : {len(contours)}")

    cv2.imwrite("Day009/contour_sample.jpg", result)

    cv2.imshow("Original", image)
    cv2.imshow("Edge", edge)
    cv2.imshow("Contour", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()