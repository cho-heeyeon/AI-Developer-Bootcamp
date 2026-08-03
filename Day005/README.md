# Day005 - OpenCV 이미지 흑백 변환

## 과제 목표

OpenCV로 이미지 파일을 불러오고,
컬러 이미지를 흑백 이미지로 변환하여 저장한다.

## 폴더 구조

- main.py: 이미지 읽기, 변환, 저장 프로그램
- sample.jpg: 원본 이미지
- gray_sample.jpg: 흑백 변환 결과
- README.md: 과제 설명

## 구현 내용

1. OpenCV 라이브러리 불러오기
2. cv2.imread()로 이미지 읽기
3. 이미지 불러오기 성공 여부 확인
4. cv2.cvtColor()로 흑백 변환
5. 원본 이미지와 흑백 이미지 크기 출력
6. cv2.imwrite()로 결과 저장
7. cv2.imshow()로 이미지 표시
8. 키 입력 후 이미지 창 종료

## 실행 방법

```bash
python Day005/main.py