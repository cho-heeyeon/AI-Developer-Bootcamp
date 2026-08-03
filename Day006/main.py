import cv2


def load_image(file_path):
    """이미지 파일을 읽어 반환합니다."""
    image = cv2.imread(file_path)

    if image is None:
        print(f"이미지를 불러올 수 없습니다: {file_path}")
        return None

    return image


def show_image_information(image):
    """이미지 크기와 채널 정보를 출력합니다."""
    height, width, channels = image.shape

    print("이미지 정보")
    print("-" * 30)
    print(f"가로 크기: {width} 픽셀")
    print(f"세로 크기: {height} 픽셀")
    print(f"채널 수: {channels}개")


def show_pixel_value(image, x, y):
    """지정한 좌표의 BGR 픽셀값을 출력합니다."""
    height, width, _ = image.shape

    if x < 0 or x >= width or y < 0 or y >= height:
        print("지정한 좌표가 이미지 범위를 벗어났습니다.")
        return

    blue, green, red = image[y, x]

    print("-" * 30)
    print(f"선택 좌표: x={x}, y={y}")
    print(f"Blue 값: {blue}")
    print(f"Green 값: {green}")
    print(f"Red 값: {red}")


def show_gray_pixel_value(image, x, y):
    """컬러 이미지를 흑백으로 변환하고 밝기값을 출력합니다."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = gray_image[y, x]

    print("-" * 30)
    print(f"흑백 밝기값: {brightness}")
    print("0에 가까울수록 검은색, 255에 가까울수록 흰색입니다.")


def main():
    file_path = "Day006/sample.jpg"

    image = load_image(file_path)

    if image is None:
        return

    show_image_information(image)

    x = 400
    y = 250

    show_pixel_value(image, x, y)
    show_gray_pixel_value(image, x, y)

    cv2.circle(image, (x, y), 8, (0, 0, 255), -1)

    cv2.imshow("Pixel Position", image)

    print("-" * 30)
    print("이미지 창에서 아무 키나 누르면 종료됩니다.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()