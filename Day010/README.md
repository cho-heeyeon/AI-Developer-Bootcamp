# Day010 - 가장 큰 Contour 선택

## 과제 목표

이미지에서 여러 개의 Contour를 검출하고,
면적이 가장 큰 Contour 하나를 선택한다.

## 사용 기술

- Python
- OpenCV
- cv2.findContours()
- cv2.contourArea()
- max()
- cv2.drawContours()

## 처리 과정

1. 이미지 경로설정
2. 이미지 읽기
3. 원본 이미지 복사
4. 흑백 변환
5. Gaussian Blur 적용
6. Canny Edge Detection
7. Contour 검출
8. Contour 면적 계산
9. Contour 면적 비교
10. 가장 큰 Contour 선택
11. 결과 이미지 저장

## 핵심 코드

```python
largest_contour = max(contours, key=cv2.contourArea)