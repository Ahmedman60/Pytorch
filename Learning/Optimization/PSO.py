# class Solution(object):
#     def maxSubArray(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: Tuple[int, int, int]
#         """
#         current_sum = 0
#         max_sum = float("-inf")
#         start = 0
#         end = 0
#         temp_start = 0

#         for right in range(len(nums)):

#             if current_sum < 0:
#                 current_sum = 0
#                 start = right

#             current_sum += nums[right]

#             if current_sum > max_sum:
#                 max_sum = current_sum
#                 end = right

#         return max_sum, start, end


# # Example Usage:
# sol = Solution()
# nums = [-2, -3, 4]
# print(sol.maxSubArray(nums))


# frozenset are awesome because they are immutable and hashable.
# class Solution(object):
#     def groupAnagrams(self, strs):
#         """
#         :type strs: List[str]
#         :rtype: List[List[str]]
#         """
#         hash_map = {}
#         for word in strs:
#             # sorted returns a list of characters, so we need to join them to make a string.
#             key = ''.join(sorted(word))
#             if hash_map.get(key) is None:
#                 hash_map[key] = [word]
#             else:
#                 hash_map[key] += [word]  # this is same as append.

#         return list(hash_map.values())


# # Example Usage:
# sol = Solution()
# # strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
# strs = ["ddddddddddg", "dgggggggggg"]

# print(sol.groupAnagrams(strs))


from collections import defaultdict


def groupAnagrams(strs):
    hash_map = defaultdict(list)

    for word in strs:
        count = [0] * 26  # start with 0s for each letter

        for c in word:
            count[ord(c) - ord('a')] += 1

        key = tuple(count)  # must be immutable to use as dict key
        hash_map[key].append(word)

    return list(hash_map.values())


# Absolutely! Let’s walk through the **character frequency technique** in a simple and visual way so you really get what’s happening and why it works for **grouping anagrams**.

# ---

# ## 💡 Why Character Frequency Works

# Two strings are **anagrams** if:
# - They contain **the exact same letters**
# - **Each letter appears the same number of times**

# 👉 So instead of sorting, we can just **count how many times each letter appears** in each word.

# ---

# ## 🔢 Character Frequency Representation

# Imagine you're working with lowercase English letters only.

# You create a **list of length 26** (one for each letter from `'a'` to `'z'`), and for each character in the word, you **increment the count** at the corresponding index.

# Example:

# ### Word: `"eat"`

# - `'e'` → index 4
# - `'a'` → index 0
# - `'t'` → index 19

# So the count array becomes:

# ```python
# [1, 0, 0, 0, 1, 0, ..., 1, ..., 0]
# # indexes 0 = a, 4 = e, 19 = t
# ```

# This becomes the **key** for grouping. Any other word with the same letter counts (like `"tea"` or `"ate"`) will produce **exactly the same count array** → so they go in the same group.

# ---

# ## ✅ Code Walkthrough

# ```python
# from collections import defaultdict

# def groupAnagrams(strs):
#     hash_map = defaultdict(list)

#     for word in strs:
#         count = [0] * 26  # start with 0s for each letter

#         for c in word:
#             count[ord(c) - ord('a')] += 1  # convert char to index

#         key = tuple(count)  # must be immutable to use as dict key
#         hash_map[key].append(word)

#     return list(hash_map.values())
# ```

# ---

# ## 🧠 Why `tuple(count)` is used

# - `list` is **not hashable**, so it can’t be used as a dictionary key.
# - `tuple` is immutable and hashable — perfect as a unique representation of each anagram group.

# ---

# ## 📊 Real Example

# Input:
# ```python
# ["eat", "tea", "tan", "ate", "nat", "bat"]
# ```

# Character frequency keys:

# | Word | Frequency Tuple (non-zero counts only for clarity)       | Group      |
# |------|----------------------------------------------------------|------------|
# | eat  | `('a':1, 'e':1, 't':1)` → `(1,0,0,0,1,...,1...)`         | Group A    |
# | tea  | same as above                                            | Group A    |
# | ate  | same as above                                            | Group A    |
# | tan  | `'a':1, 'n':1, 't':1` → different key                    | Group B    |
# | nat  | same as above                                            | Group B    |
# | bat  | `'a':1, 'b':1, 't':1` → different key                    | Group C    |

# So output:

# ```python
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
# ```

# ---

# ## ✅ Summary

# | Aspect                     | Character Frequency Approach      |
# |---------------------------|-----------------------------------|
# | ✅ Fast                   | O(N*K), no sorting involved       |
# | ✅ Accurate               | Accounts for letter **counts**    |
# | ✅ Hashable key           | Using `tuple(count)`              |
# | 🔧 Needs lowercase input | Assumes only `'a'` to `'z'`        |

# ---

# Would you like a version that supports **unicode or mixed-case letters** too?
