import sys

# def main(lines):
    # このコードは標準入力と標準出力を用いたサンプルコードです。
    # このコードは好きなように編集・削除してもらって構いません。
    # ---
    # This is a sample code to use stdin and stdout.
    # Edit and remove this code as you like.

    # for i, v in enumerate(lines):
        # print("line[{0}]: {1}".format(i, v))

def greenness_on_other_side(greenness):
    n = len(greenness)
    left_max = [0] * n
    right_max = [0] * n
    left_max[0] = greenness[0]
    right_max[n-1] = greenness[n-1]

    for i in range(1, n):
        left_max[i] = max(left_max[i-1], greenness[i])
        right_max[n-1-i] = max(right_max[n-i], greenness[n-1-i])

    greenness_other_side = []
    for i in range(n):
        if i == 0:
            greenness_other_side.append(right_max[i+1])
        elif i == n-1:
            greenness_other_side.append(left_max[i-1])
        else:
            greenness_other_side.append(max(left_max[i-1], right_max[i+1]))

    return greenness_other_side


if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.rstrip('\r\n'))
    n = int(lines[0])
    arr = list(map(int, lines[1].split()))

    if n < 2 or n > 100000:
        print("Error")
    arr_length = len(arr)
    if arr_length < 2 or arr_length > 100000:
        print("Error")
    for num in arr:
        if num < -1000000000 or num > 1000000000:
            print("Error")


    result = greenness_on_other_side(arr)
    for line in result:
        print(line)