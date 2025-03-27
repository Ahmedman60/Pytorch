

class Solution(object):
    def longestOnes(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        max_len = 0
        left = 0
        zero_count = 0

        for right in range(len(nums)):
            k -= 1 - nums[right]
            if k < 0:  #
                if nums[left] == 0:
                    # giveback the onces if 0's if 1 we don't do anything
                    k += 1
                left += 1
            else:
                max_len = max(max_len, right - left + 1)

        return max_len


print(Solution().longestOnes(
    [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1], k=3))
