# class Solution(object):
#     def longestSubarray(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         if nums.count(1) == len(nums):
#             return len(nums)-1

#         max_len = 0
#         left = 0
#         delete = False
#         for right in range(len(nums)):
#             if delete:
#                 left += 1
#                 delete = False

#             if nums[right] == 0:
#                 delete = True

#             if delete:
#                 max_len = max(max_len, (right-left+1)-1)
#             else:
#                 max_len = max(max_len, (right-left+1))

#         return max_len


# print(Solution().longestSubarray([1, 1, 1]))  # 3


# class Solution(object):
#     def longestSubarray(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         if nums.count(1) == len(nums):
#             return len(nums)-1

#         left = 0
#         current_len = 0
#         delete = False
#         max_len = 0
#         for right in range(len(nums)):

#             if nums[right] == 0:
#                 delete = True
#                 # for ones
#                 while nums[left] == 1:
#                     left += 1
#                 # for the zero
#                 left += 1
#             if delete == True:
#                 current_len += right-left+1
#             else:
#                 current_len = right-left+1

#             delete = False
#             max_len = max(max_len, current_len)


class Solution(object):
    def longestSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        zero_index = -1
        max_len = 0
        # [0, 1, 1, 1, 0, 1, 1, 0, 1]
        for right in range(len(nums)):

            if nums[right] == 0:
                left = zero_index + 1
                zero_index = right

            max_len = max(max_len, right - left)

        return max_len


print(Solution().longestSubarray([0, 1, 1, 1, 0, 1, 1, 0, 1]))  # 3
