def find_k_smallest(arr, k):
    while True:
        pivot = arr[-1]  # Choose the last element as the pivot
        left = [x for x in arr if x < pivot]  # Elements < pivot
        right = [x for x in arr if x > pivot]  # Elements > pivot

        # Count the pivot itself
        # incase there is multiple element with same value as pivot
        pivot_count = len(arr) - len(left) - len(right)

        if len(left) == k:  # Found exactly k smallest elements
            return left
        elif len(left) + pivot_count >= k:  # Include the pivot as part of the result
            return (left + [pivot] * (k - len(left)))[:k]
        else:  # Not enough elements, move to the right part
            k -= (len(left) + pivot_count)  # Adjust k
            arr = right


arr = [7, 2, 4, 1, 8, 6, 3, 5, 1, 1, 4, 4, 4]
k = 3
result = find_k_smallest(arr, k)
print(result)  # Output: [2, 1, 3] (order may vary)


# This code isn't save to use it drop in some usage cases.
