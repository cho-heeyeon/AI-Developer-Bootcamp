# Day004 - CSV 측정값 자동 판정

## 과제 목표

CSV 파일에 저장된 샤프트 외경 측정값을 Python으로 읽고,
규격 범위에 따라 OK와 NG를 자동 판정한다.

## 폴더 구조

- main.py: CSV 읽기 및 측정값 판정
- measurements.csv: 외경 측정 데이터
- README.md: 과제 설명

## 규격 조건

- 하한: 19.95 mm
- 상한: 20.05 mm

## 구현 내용

1. csv 모듈로 CSV 파일 읽기
2. 문자열을 int와 float로 변환
3. 리스트에 측정 데이터 저장
4. for 반복문으로 측정값 순회
5. if 조건문으로 OK/NG 판정
6. 전체, OK, NG 개수 출력
7. 파일 및 데이터 오류 예외 처리

## 실행 방법

```bash
python Day004/main.py