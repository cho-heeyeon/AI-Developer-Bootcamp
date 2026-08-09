YOLO labels 생성 - 검증
Day021/
├─ images/
├─ labels/
├─ verified/
│  ├─ sample_03_verified.jpg
│  ├─ sample_04_verified.jpg
│  ├─ sample_06_verified.jpg
│  ├─ sample_07_verified.jpg
│  ├─ sample_N01_verified.jpg
│  └─ sample_N02_verified.jpg
│
├─ make_labels.py
├─ verify_labels.py
├─ dataset.yaml
└─ README.md

특히 확인할 것은 세 가지입니다.

박스가 샤프트 전체가 아니라 실제 측정할 외경 구간만 포함하는가
6개 이미지에서 박스 위치가 같은 종류의 단차/외경부에 놓이는가
위·아래 외곽선을 충분히 포함해서 이후 OpenCV Sub-pixel 측정이 가능한가

 - 세가지 전부 확인 했습니다. 준비 완료입니다