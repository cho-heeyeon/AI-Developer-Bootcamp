import cv2

# 전역 변수
image = None


def mouse_callback(event, x, y, flags, param):
    global image

    if event == cv2.EVENT_LBUTTONDOWN:

        blue, green, red = image[y, x]

        print("-" * 35)
        print(f"클릭한 좌표 : ({x}, {y})")
        print(f"B : {blue}")
        print(f"G : {green}")
        print(f"R : {red}")

        image_copy = image.copy()

        cv2.circle(image_copy, (x, y), 6, (0, 0, 255), -1)

        cv2.imshow("Sample Image", image_copy)


def main():

    global image

    image = cv2.imread("Day007/sample.jpg")

    if image is None:
        print("이미지를 읽을 수 없습니다.")
        return

    cv2.imshow("Sample Image", image)

    cv2.setMouseCallback("Sample Image", mouse_callback)

    print("이미지를 클릭해 보세요.")
    print("ESC 키를 누르면 종료됩니다.")

    while True:

        key = cv2.waitKey(1)

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()