# Day012 - ROI 추출

## 과제 목표

Bounding Rectangle를 이용하여
샤프트 부분만 추출한다.

## 사용 기술

- Python
- OpenCV
- cv2.boundingRect()
- ROI(Image Slice)
- cv2.imwrite()

## 처리 순서

Image

↓

Gray

↓

Edge

↓

Contour

↓

Bounding Rectangle

↓

ROI 추출

↓

ROI 저장

## 실행 결과

roi_sample.jpg 생성 완료

## 배운 점

ROI를 이용하면
이미지 전체가 아니라
샤프트 영역만 분석할 수 있다.