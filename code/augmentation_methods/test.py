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

# Example usage:
greenness = [3, 2, 5, 4, 1]
result = greenness_on_other_side(greenness)
print("Greenness on the other side for each garden:", result)
for i in result:
    print(i)
