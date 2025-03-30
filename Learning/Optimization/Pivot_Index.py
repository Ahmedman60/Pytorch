# class Solution(object):
#     def pivotIndex(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         prefix = [0]*len(nums)
#         sufix = [0]*len(nums)
#         current_sum_Left = 0
#         current_sum_right = 0
#         for i in range(len(nums)):

#             current_sum_Left += nums[i]
#             prefix[i] += current_sum_Left

#             current_sum_right += nums[(len(nums)-1)-i]
#             sufix[(len(nums)-1)-i] += current_sum_right

#         for i in range(len(nums)):
#             if prefix[i] == sufix[i]:
#                 return i

#         return -1

# # Test case


# # Expected output: 3 (pivot index is 3, as the sum of elements to the left is equal to the sum of elements to the right)
# print(Solution().pivotIndex([1, 7, 3, 6, 5, 6]))


class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        summation = sum(nums)

        left_sum = 0

        for i in range(len(nums)):

            # the right summation. this will always remove the extra element.
            summation -= nums[i]

            if left_sum == summation:
                return i

            left_sum += nums[i]  # the left summation

        return -1


class Solution(object):
    def pivotIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        total_sum = sum(nums)
        left_sum = 0

        for i, num in enumerate(nums):

            # subtact the left summation and the currect pivot index.
            # https://www.youtube.com/watch?v=u89i60lYx8U&ab_channel=NeetCode
            if left_sum == total_sum - left_sum - num:
                return i
            left_sum += num

        return -1
