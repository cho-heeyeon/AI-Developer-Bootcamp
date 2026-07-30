# 샤프트 외경 측정값
measurements = [20.01, 19.98, 20.07, 19.94, 20.00]

# 규격 하한과 상한
lower_limit = 19.95
upper_limit = 20.05

# OK와 NG 개수를 저장할 변수
ok_count = 0
ng_count = 0

print("샤프트 외경 측정 결과")
print("-" * 35)

# 측정값을 하나씩 꺼내서 검사
for index, value in enumerate(measurements, start=1):

    if lower_limit <= value <= upper_limit:
        result = "OK"
        ok_count = ok_count + 1
    else:
        result = "NG"
        ng_count = ng_count + 1

    print(f"{index}번 측정값: {value:.2f} mm → {result}")

print("-" * 35)
print(f"전체 측정 개수: {len(measurements)}개")
print(f"OK 개수: {ok_count}개")
print(f"NG 개수: {ng_count}개")