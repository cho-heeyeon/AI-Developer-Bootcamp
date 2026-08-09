[Day024-1 원인 분석]

1. YOLO는 측정 ROI 탐색에 사용한다.
2. YOLO Bounding Box 자체를 외경 측정값으로 사용하지 않는다.
3. ROI 내부에 반사광 및 가공면 Edge가 여러 개 존재한다.
4. 기존 Sub-pixel 알고리즘이 내부 Edge를 외곽 Edge로 오인할 수 있다.
5. 실제 정상 외경은 약 462.72 pixel로 확인되었다.
6. 따라서 OpenCV 외곽 Edge 선택 알고리즘 개선이 필요하다.
7. YOLO confidence가 낮으면 측정을 수행하지 않는다.
8. 향후 실제 원본 이미지를 추가하여 YOLO 일반화 성능을 재검증한다.