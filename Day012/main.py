import cv2
from pathlib import Path


def main():

    image_path = Path("Day011/sample.jpg")

    image = cv2.imread(str(image_path))

    if image is None:
        print("이미지를 읽을 수 없습니다.")
        return

    result = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5),0)

    edge = cv2.Canny(blur,50,150)

    contours, hierarchy = cv2.findContours(
        edge,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print("Contour 개수 :",len(contours))

    if len(contours)==0:
        print("Contour 없음")
        return

    largest = max(contours,key=cv2.contourArea)

    area = cv2.contourArea(largest)

    print("가장 큰 면적 :",area)

    x,y,w,h = cv2.boundingRect(largest)

    print("x :",x)
    print("y :",y)
    print("width :",w)
    print("height :",h)

    

    roi = image[y:y+h, x:x+w]

    output_path = Path("Day012/roi_sample.jpg")

    cv2.imwrite(str(output_path), roi)

    print("ROI 저장 완료 :", output_path)


if __name__=="__main__":
    main()