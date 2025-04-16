# # class Solution(object):
# #     def closeStrings(self, word1, word2):
# #         """
# #         :type word1: str
# #         :type word2: str
# #         :rtype: bool
# #         """

# #         freq_count1 = [0]*26
# #         freq_count2 = [0]*26
# #         for i in word1:
# #             freq_count1[ord(i)-ord('a')] += 1

# #         for i in word2:
# #             freq_count2[ord(i)-ord('a')] += 1

# #         return sorted(freq_count1) == sorted(freq_count2) and set(word1) == set(word2)


# from collections import Counter


# class Solution(object):
#     def closeStrings(self, word1, word2):
#         """
#         :type word1: str
#         :type word2: str
#         :rtype: bool
#         """
#         if len(word1) != len(word2):
#             return False

#         freq1 = Counter(word1)
#         freq2 = Counter(word2)
#         print(freq1.keys(), freq2.keys())
#         return (sorted(freq1.values()) == sorted(freq2.values())) and (freq1.keys() == freq2.keys())


# print(Solution().closeStrings("kyq", "kqy"))

# class Solution:
#     def singleNumber(self, nums):
#         nums.sort()
#         i = 0
#         while i < len(nums) - 1:
#             if nums[i] != nums[i + 1]:
#                 return nums[i]
#             i += 2
#         return nums[-1]


# class Solution:
#     def singleNumber(self, nums):

#         z = nums[0]

#         for i in nums[1:]:
#             if z & i:
#                 z = z ^ i
#             else:
#                 z = z | i

#         return z


# Solution().singleNumber([2, 2, 1])


class Solution:
    def singleNumber(self, nums) -> int:
        z = nums[0]
        visited = set()
        for i in nums[1:]:
            if z & (1 << i):
                z ^= i
                visited.add(i)
            else:
                if i not in visited:
                    z |= i
        return z


print(Solution().singleNumber([2, 2, 3, 2]))
