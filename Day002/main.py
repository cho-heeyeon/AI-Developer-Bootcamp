def analyze_numbers(numbers):
    """숫자 목록의 기본 통계값을 계산한다."""

    if not numbers:
        return None

    total = sum(numbers)
    average = total / len(numbers)
    maximum = max(numbers)
    minimum = min(numbers)
    sorted_numbers = sorted(numbers)

    return total, average, maximum, minimum, sorted_numbers


numbers = [35, 12, 48, 7, 26]

result = analyze_numbers(numbers)

if result is None:
    print("분석할 숫자가 없습니다.")
else:
    total, average, maximum, minimum, sorted_numbers = result

    print("원본 숫자 :", numbers)
    print("합계 :", total)
    print("평균 :", average)
    print("최댓값 :", maximum)
    print("최솟값 :", minimum)
    print("오름차순 :", sorted_numbers)