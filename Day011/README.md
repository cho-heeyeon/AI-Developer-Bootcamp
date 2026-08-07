# Day011 - Bounding Rectangle

## 과제목표

Contour를 감싸는 Bounding Rectangle을 생성한다.

## 사용기술

Python

OpenCV

cv2.boundingRect()

cv2.rectangle()

## 처리순서

이미지

↓

Gray

↓

Blur

↓

Edge

↓

Contour

↓

Largest Contour

↓

Bounding Rectangle

## 핵심코드

x,y,w,h = cv2.boundingRect(contour)

## 실행결과

Contour 개수 :1015

면적 :6673.0

x :1574

y :1197

width :464

height :307

## 배운점

Bounding Rectangle은 Contour를 감싸는 최소 사각형이다.

ROI를 만드는 기초가 된다.