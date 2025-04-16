# class Solution(object):
# #     def isAnagram(self, s, t):
# #         """
# #         :type s: str
# #         :type t: str
# #         :rtype: bool
# #         """
# #         # frequency count
# #         hash_map1 = [0]*26
# #         hash_map2 = [0]*26
# #         for i in range(len(s)):
# #             hash_map1[ord(s[i])-ord('a')] += 1

# #         for i in range(len(t)):
# #             hash_map2[ord(t[i])-ord('a')] += 1

# #         return hash_map1 == hash_map2


# # sol = Solution()
# # # print(sol.isAnagram("anagram", "nagaram"))  # True
# # # print(sol.isAnagram("rat", "car"))  # False
# # print(sol.isAnagram("a", "ab"))  # False


# hash_map = {'a': [], 'b': [], 'c': []}
# print(hash_map.values())
# print(any(hash_map.values()))

# if any(hash_map.values()):  # anyone has value
#     print("False")
# else:
#     print("False")