# Day008 - OpenCV Canny Edge Detection

## 과제 목표

OpenCV의 Canny 알고리즘을 이용하여 이미지에서 샤프트와
배경 사이의 경계선을 검출한다.

## 구현 내용

- 이미지 불러오기
- 컬러 이미지를 흑백으로 변환
- Gaussian Blur로 노이즈 감소
- Canny Edge Detection 적용
- 원본, 흑백, Blur, Edge 이미지 출력
- 에지 결과 이미지 저장

## 실행 방법

```bash
python Day008/main.py