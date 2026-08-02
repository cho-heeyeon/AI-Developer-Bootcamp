import csv


LOWER_LIMIT = 19.95
UPPER_LIMIT = 20.05


def load_measurements(file_path):
    """CSV 파일에서 외경 측정값을 읽어 리스트로 반환합니다."""
    measurements = []

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)
            print(reader.fieldnames)

            for row in reader:
                number = int(row["number"])
                diameter = float(row["diameter"])

                measurements.append(
                    {
                        "number": number,
                        "diameter": diameter
                    }
                )

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return []

    except (ValueError, KeyError) as error:
        print(f"CSV 데이터 형식에 문제가 있습니다: {error}")
        return []

    return measurements


def inspect_measurements(measurements):
    """측정값을 규격과 비교하여 OK/NG를 판정합니다."""
    ok_count = 0
    ng_count = 0

    print("샤프트 외경 측정 결과")
    print("-" * 40)

    for item in measurements:
        number = item["number"]
        diameter = item["diameter"]

        if LOWER_LIMIT <= diameter <= UPPER_LIMIT:
            result = "OK"
            ok_count += 1
        else:
            result = "NG"
            ng_count += 1

        print(f"{number}번: {diameter:.2f} mm → {result}")

    print("-" * 40)
    print(f"전체 측정 개수: {len(measurements)}개")
    print(f"OK 개수: {ok_count}개")
    print(f"NG 개수: {ng_count}개")


def main():
    file_path = "Day004/measurements.csv"

    measurements = load_measurements(file_path)

    if not measurements:
        print("분석할 측정값이 없습니다.")
        return

    inspect_measurements(measurements)


if __name__ == "__main__":
    main()