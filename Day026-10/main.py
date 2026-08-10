import cv2
import csv
import numpy as np

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path("Day026-10")
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = Path(
    "runs/detect/Day023/runs/"
    "measurement_roi_aug-2/weights/best.pt"
)


SAMPLES = [
    "sample_03",
    "sample_04",
    "sample_06",
    "sample_07",
    "sample_N01",
    "sample_N02",
]


# ============================================================
# 2. Continuity 설정
# ============================================================

# 이전 x 위치와 비교했을 때
# 상/하 Edge가 이 값보다 크게 이동하면 점프 후보
MAX_EDGE_JUMP = 12.0

# 외경 자체가 이전 외경에서 이 이상 변하면 점프 후보
MAX_DIAMETER_JUMP = 15.0

# 후보 gradient 최소 상대 강도
STRENGTH_RATIO = 0.40

# 측정 x 개수
NUM_MEASURE_POINTS = 31


# ============================================================
# 3. YOLO
# ============================================================

model = YOLO(
    str(MODEL_PATH)
)


# ============================================================
# 4. Sub-pixel 보간
# ============================================================

def subpixel_edge(
    abs_gradient,
    index
):

    if index <= 0:
        return float(index)

    if index >= len(abs_gradient) - 1:
        return float(index)


    g1 = float(
        abs_gradient[index - 1]
    )

    g2 = float(
        abs_gradient[index]
    )

    g3 = float(
        abs_gradient[index + 1]
    )


    denominator = (
        g1
        - 2.0 * g2
        + g3
    )


    if abs(denominator) < 1e-12:
        return float(index)


    offset = (
        0.5
        * (g1 - g3)
        / denominator
    )


    offset = np.clip(
        offset,
        -1.0,
        1.0
    )


    return (
        float(index)
        + float(offset)
    )


# ============================================================
# 5. 한 x 위치에서 Edge 후보 여러 개 생성
# ============================================================

def generate_edge_candidates(
    gray,
    x
):

    strip_x1 = max(
        0,
        x - 2
    )

    strip_x2 = min(
        gray.shape[1],
        x + 3
    )


    profile = np.mean(
        gray[
            :,
            strip_x1:strip_x2
        ],
        axis=1
    ).astype(np.float32)


    gradient = np.gradient(
        profile
    )


    abs_gradient = np.abs(
        gradient
    )


    h = len(profile)


    # --------------------------------------------------------
    # 상단 / 하단 검색 영역
    # --------------------------------------------------------

    upper_end = int(
        h * 0.46
    )

    lower_start = int(
        h * 0.54
    )


    upper_abs = abs_gradient[
        :upper_end
    ]

    lower_abs = abs_gradient[
        lower_start:
    ]


    if len(upper_abs) == 0:
        return []

    if len(lower_abs) == 0:
        return []


    # --------------------------------------------------------
    # 강한 Edge 후보
    # --------------------------------------------------------

    upper_threshold = np.percentile(
        upper_abs,
        78
    )

    lower_threshold = np.percentile(
        lower_abs,
        78
    )


    upper_candidates = np.where(
        upper_abs >= upper_threshold
    )[0]


    lower_candidates_local = np.where(
        lower_abs >= lower_threshold
    )[0]


    lower_candidates = (
        lower_candidates_local
        + lower_start
    )


    candidate_pairs = []


    # --------------------------------------------------------
    # 상/하 Edge 조합 생성
    # --------------------------------------------------------

    for top_idx in upper_candidates:

        top_grad = float(
            gradient[top_idx]
        )


        for bottom_idx in lower_candidates:

            bottom_grad = float(
                gradient[bottom_idx]
            )


            # 실제 외곽선은 gradient 방향이 반대
            if (
                top_grad
                * bottom_grad
                >= 0
            ):
                continue


            diameter_integer = (
                bottom_idx
                - top_idx
            )


            # 지나치게 작거나 큰 후보 제거
            if diameter_integer < h * 0.25:
                continue

            if diameter_integer > h * 0.60:
                continue


            strength = (
                abs(top_grad)
                + abs(bottom_grad)
            )


            top_sub = subpixel_edge(
                abs_gradient,
                int(top_idx)
            )


            bottom_sub = subpixel_edge(
                abs_gradient,
                int(bottom_idx)
            )


            diameter = (
                bottom_sub
                - top_sub
            )


            candidate_pairs.append(
                {
                    "top":
                        float(top_sub),

                    "bottom":
                        float(bottom_sub),

                    "diameter":
                        float(diameter),

                    "strength":
                        float(strength),

                    "top_grad":
                        float(top_grad),

                    "bottom_grad":
                        float(bottom_grad),
                }
            )


    if len(candidate_pairs) == 0:
        return []


    # --------------------------------------------------------
    # 너무 약한 후보 제거
    # --------------------------------------------------------

    max_strength = max(
        c["strength"]
        for c in candidate_pairs
    )


    strength_limit = (
        max_strength
        * STRENGTH_RATIO
    )


    candidate_pairs = [
        c
        for c in candidate_pairs
        if c["strength"]
        >= strength_limit
    ]


    return candidate_pairs


# ============================================================
# 6. 첫 번째 기준 Edge 선택
# ============================================================

def select_initial_candidate(
    candidates
):

    if len(candidates) == 0:
        return None


    diameters = np.array(
        [
            c["diameter"]
            for c in candidates
        ],
        dtype=np.float64
    )


    median_diameter = float(
        np.median(diameters)
    )


    # 강도와 median 근접성을 함께 사용
    best = min(
        candidates,
        key=lambda c:
            abs(
                c["diameter"]
                - median_diameter
            )
            - 0.01 * c["strength"]
    )


    return best


# ============================================================
# 7. Continuity 기반 후보 선택
# ============================================================

def select_continuous_candidate(
    candidates,
    previous
):

    if len(candidates) == 0:
        return None, "NO_CANDIDATE"


    if previous is None:

        initial = select_initial_candidate(
            candidates
        )

        return (
            initial,
            "INITIAL"
        )


    scored = []


    for candidate in candidates:

        top_jump = abs(
            candidate["top"]
            - previous["top"]
        )


        bottom_jump = abs(
            candidate["bottom"]
            - previous["bottom"]
        )


        diameter_jump = abs(
            candidate["diameter"]
            - previous["diameter"]
        )


        # ----------------------------------------------------
        # 사전 차단
        # ----------------------------------------------------

        if top_jump > MAX_EDGE_JUMP:
            continue

        if bottom_jump > MAX_EDGE_JUMP:
            continue

        if diameter_jump > MAX_DIAMETER_JUMP:
            continue


        # ----------------------------------------------------
        # 작은 이동을 우선 선택
        # ----------------------------------------------------

        continuity_cost = (
            top_jump
            + bottom_jump
            + 0.5 * diameter_jump
        )


        # 강한 Edge에 약간의 보너스
        score = (
            continuity_cost
            - 0.01 * candidate["strength"]
        )


        scored.append(
            (
                score,
                candidate
            )
        )


    # --------------------------------------------------------
    # 연속성 조건을 만족하는 후보가 없음
    # --------------------------------------------------------

    if len(scored) == 0:

        return (
            None,
            "JUMP_REJECTED"
        )


    scored.sort(
        key=lambda item:
            item[0]
    )


    return (
        scored[0][1],
        "TRACKED"
    )


# ============================================================
# 8. 한 샘플 측정
# ============================================================

def analyze_sample(
    sample_name
):

    image_path = (
        IMAGE_DIR
        / f"{sample_name}.jpg"
    )


    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            sample_name,
            ": 이미지 읽기 실패"
        )

        return []


    # ========================================================
    # YOLO ROI
    # ========================================================

    predictions = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        verbose=False
    )


    best_box = None
    best_conf = 0.0


    for prediction in predictions:

        if prediction.boxes is None:
            continue


        for box in prediction.boxes:

            confidence = float(
                box.conf[0]
            )


            if confidence > best_conf:

                best_conf = confidence
                best_box = box


    if best_box is None:

        print(
            sample_name,
            ": YOLO 검출 실패"
        )

        return []


    image_h, image_w = (
        image.shape[:2]
    )


    x1, y1, x2, y2 = (
        best_box.xyxy[0]
        .cpu()
        .numpy()
        .astype(int)
    )


    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        image_w,
        x2
    )

    y2 = min(
        image_h,
        y2
    )


    box_h = (
        y2 - y1
    )


    vertical_margin = int(
        box_h * 0.40
    )


    ey1 = max(
        0,
        y1 - vertical_margin
    )


    ey2 = min(
        image_h,
        y2 + vertical_margin
    )


    roi = image[
        ey1:ey2,
        x1:x2
    ].copy()


    if roi.size == 0:

        print(
            sample_name,
            ": ROI 생성 실패"
        )

        return []


    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )


    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    roi_h, roi_w = (
        blur.shape
    )


    # ========================================================
    # 31개 x 위치
    # ========================================================

    x_positions = np.linspace(
        int(roi_w * 0.18),
        int(roi_w * 0.82),
        NUM_MEASURE_POINTS
    ).astype(int)


    # --------------------------------------------------------
    # 중앙에서 시작
    #
    # 왼쪽→오른쪽 단방향보다
    # 중앙에서 양쪽으로 추적하는 것이 안정적
    # --------------------------------------------------------

    center_index = (
        len(x_positions) // 2
    )


    selected = {
        center_index: None
    }


    status_map = {}


    # ========================================================
    # 중앙 기준점
    # ========================================================

    center_x = int(
        x_positions[
            center_index
        ]
    )


    center_candidates = (
        generate_edge_candidates(
            blur,
            center_x
        )
    )


    center_edge = (
        select_initial_candidate(
            center_candidates
        )
    )


    if center_edge is None:

        print(
            sample_name,
            ": 중앙 Edge 검출 실패"
        )

        return []


    selected[
        center_index
    ] = center_edge


    status_map[
        center_index
    ] = "INITIAL"


    # ========================================================
    # 중앙 → 오른쪽 추적
    # ========================================================

    previous = center_edge


    for i in range(
        center_index + 1,
        len(x_positions)
    ):

        x = int(
            x_positions[i]
        )


        candidates = (
            generate_edge_candidates(
                blur,
                x
            )
        )


        edge, status = (
            select_continuous_candidate(
                candidates,
                previous
            )
        )


        selected[i] = edge
        status_map[i] = status


        # 성공했을 때만 previous 갱신
        if edge is not None:
            previous = edge


    # ========================================================
    # 중앙 → 왼쪽 추적
    # ========================================================

    previous = center_edge


    for i in range(
        center_index - 1,
        -1,
        -1
    ):

        x = int(
            x_positions[i]
        )


        candidates = (
            generate_edge_candidates(
                blur,
                x
            )
        )


        edge, status = (
            select_continuous_candidate(
                candidates,
                previous
            )
        )


        selected[i] = edge
        status_map[i] = status


        if edge is not None:
            previous = edge


    # ========================================================
    # 결과 정리
    # ========================================================

    records = []


    for i, x in enumerate(
        x_positions
    ):

        edge = selected.get(
            i
        )


        status = status_map.get(
            i,
            "UNKNOWN"
        )


        if edge is None:

            records.append(
                {
                    "sample":
                        sample_name,

                    "point":
                        i + 1,

                    "x":
                        int(x),

                    "top_y":
                        "",

                    "bottom_y":
                        "",

                    "diameter":
                        "",

                    "tracking_status":
                        status,

                    "confidence":
                        best_conf,
                }
            )

            continue


        records.append(
            {
                "sample":
                    sample_name,

                "point":
                    i + 1,

                "x":
                    int(x),

                "top_y":
                    edge["top"],

                "bottom_y":
                    edge["bottom"],

                "diameter":
                    edge["diameter"],

                "tracking_status":
                    status,

                "confidence":
                    best_conf,
            }
        )


    # ========================================================
    # 통계
    # ========================================================

    valid_records = [
        r
        for r in records
        if isinstance(
            r["diameter"],
            float
        )
    ]


    rejected_records = [
        r
        for r in records
        if r["tracking_status"]
        == "JUMP_REJECTED"
    ]


    if len(valid_records) > 0:

        diameters = np.array(
            [
                r["diameter"]
                for r in valid_records
            ],
            dtype=np.float64
        )


        median_px = float(
            np.median(
                diameters
            )
        )


        mean_px = float(
            np.mean(
                diameters
            )
        )


        std_px = float(
            np.std(
                diameters
            )
        )


    else:

        median_px = np.nan
        mean_px = np.nan
        std_px = np.nan


    # ========================================================
    # 시각화
    # ========================================================

    debug = roi.copy()


    for record in records:

        x = int(
            record["x"]
        )


        if isinstance(
            record["diameter"],
            float
        ):

            top_y = int(
                round(
                    record["top_y"]
                )
            )


            bottom_y = int(
                round(
                    record["bottom_y"]
                )
            )


            # 정상 추적 = 초록
            cv2.line(
                debug,
                (x, top_y),
                (x, bottom_y),
                (0, 255, 0),
                1
            )


            cv2.circle(
                debug,
                (x, top_y),
                2,
                (0, 255, 255),
                -1
            )


            cv2.circle(
                debug,
                (x, bottom_y),
                2,
                (0, 255, 255),
                -1
            )


        else:

            # Reject 위치 = 빨간 세로 표시
            cv2.line(
                debug,
                (x, 0),
                (x, roi_h - 1),
                (0, 0, 255),
                1
            )


    cv2.putText(
        debug,
        sample_name,
        (5, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 255),
        1
    )


    cv2.putText(
        debug,
        f"Median={median_px:.2f}px",
        (5, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1
    )


    cv2.putText(
        debug,
        f"STD={std_px:.3f}px",
        (5, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1
    )


    cv2.putText(
        debug,
        f"Rejected={len(rejected_records)}",
        (5, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 255),
        1
    )


    output_path = (
        OUTPUT_DIR
        / (
            f"{sample_name}"
            "_continuity.jpg"
        )
    )


    cv2.imwrite(
        str(output_path),
        debug
    )


    # ========================================================
    # 터미널 출력
    # ========================================================

    print()
    print(
        "========================================"
    )

    print(
        "Sample :",
        sample_name
    )

    print(
        "YOLO confidence :",
        best_conf
    )

    print(
        "전체 측정 위치 :",
        NUM_MEASURE_POINTS
    )

    print(
        "유효 측정 :",
        len(valid_records)
    )

    print(
        "Jump Reject :",
        len(rejected_records)
    )

    print(
        "Median(px) :",
        median_px
    )

    print(
        "Mean(px) :",
        mean_px
    )

    print(
        "STD(px) :",
        std_px
    )


    return records


# ============================================================
# 9. 전체 샘플 실행
# ============================================================

print()
print(
    "========================================"
)

print(
    "Day026-10"
)

print(
    "Edge Continuity Tracking"
)

print(
    "========================================"
)


all_records = []


for sample in SAMPLES:

    sample_records = (
        analyze_sample(
            sample
        )
    )


    all_records.extend(
        sample_records
    )


# ============================================================
# 10. CSV 저장
# ============================================================

csv_path = (
    OUTPUT_DIR
    / "continuity_tracking_results.csv"
)


fieldnames = [
    "sample",
    "point",
    "x",
    "top_y",
    "bottom_y",
    "diameter",
    "tracking_status",
    "confidence",
]


with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )


    writer.writeheader()


    for record in all_records:

        writer.writerow(
            record
        )


# ============================================================
# 11. 전체 결과 요약
# ============================================================

print()
print(
    "========================================"
)

print(
    "Day026-10 전체 결과"
)

print(
    "========================================"
)


for sample in SAMPLES:

    sample_records = [
        r
        for r in all_records
        if r["sample"] == sample
    ]


    valid = [
        r
        for r in sample_records
        if isinstance(
            r["diameter"],
            float
        )
    ]


    rejected = [
        r
        for r in sample_records
        if r["tracking_status"]
        == "JUMP_REJECTED"
    ]


    if len(valid) > 0:

        values = np.array(
            [
                r["diameter"]
                for r in valid
            ]
        )


        median_value = float(
            np.median(values)
        )


        std_value = float(
            np.std(values)
        )


    else:

        median_value = np.nan
        std_value = np.nan


    print(
        sample,
        "| valid:",
        len(valid),
        "| reject:",
        len(rejected),
        "| median:",
        round(
            median_value,
            3
        ),
        "| STD:",
        round(
            std_value,
            3
        )
    )


print()
print(
    "CSV 저장 :",
    csv_path
)


print()
print(
    "========================================"
)

print(
    "Day026-10 완료"
)

print(
    "========================================"
)