# class Solution(object):
#     def lengthOfLongestSubstring(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         # pwwkew
#         seen = set()
#         left = 0
#         max_lenght = 0
#         for right in range(len(s)):

#             if s[right] not in seen:
#                 seen.add(s[right])
#             else:
#                 # remove eveything and start new substring
#                 while s[right] in seen:
#                     seen.remove(s[left])
#                     left += 1

#             seen.add(s[right])
#             max_lenght = max(max_lenght, (right-left)+1)

#         return max_lenght


# s = Solution()
# print(s.lengthOfLongestSubstring("dvdf"))


class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        seen = set()
        left = 0
        max_length = 0

        for right in range(len(s)):
            # only remove characters until you remove the first occurrence of char.
            if s[right] not in seen:
                seen.add(s[right])
            else:
                while s[right] in seen:
                    seen.remove(s[left])
                    left += 1
                seen.add(s[right])

            max_length = max(max_length, right - left + 1)

        return max_length


s = Solution().lengthOfLongestSubstring("abccccba")

print(s)
