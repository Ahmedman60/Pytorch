from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hash_map = {}
        for index, i in enumerate(nums):
            if target-i in hash_map:
                return [hash_map[target-i], index]
            hash_map[i] = index


Solution().twoSum([2, 7, 11, 15], 9)
