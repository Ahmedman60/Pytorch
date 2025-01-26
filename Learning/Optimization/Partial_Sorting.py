def find_k_smallest(arr, k):
    left, right = 0, len(arr) - 1

    while True:
        # Partition the array and get the pivot index
        pivot_index = partition(arr, left, right)

        if pivot_index == k - 1:
            # Found the correct position for the k smallest items
            return arr[:k]
        elif pivot_index > k - 1:
            # Search the left part
            right = pivot_index - 1
        else:
            # Search the right part
            left = pivot_index + 1


def partition(arr, left, right):
    pivot = arr[right]  # Choose the last element as the pivot
    i = left - 1  # Pointer for smaller elements

    for j in range(left, right):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]  # Swap smaller element to the left

    # Place the pivot in its correct position
    arr[i + 1], arr[right] = arr[right], arr[i + 1]
    return i + 1


arr = [7, 2, 1, 8, 6, 3, 5, 4]
k = 3
result = find_k_smallest(arr, k)
print(result)  # Output: [2, 1, 3] (order may vary)
