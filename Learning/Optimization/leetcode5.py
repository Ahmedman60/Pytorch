# # from collections import defaultdict
# # from typing import List
# # import pandas as pd


# # def inspect(results):
# #     result = defaultdict(list)
# #     for i in results:
# #         lhs = tuple(results[i][2][0][0])
# #         rhs = tuple(results[i][2][0][1])
# #         s = results[i][1]
# #         c = results[i][2][0][2]
# #         l = results[i][2][0][-1]
# #         result["Left Hand Side"].append(lhs)
# #         result["Right Hand Side"].append(rhs)
# #         result["Support"].append(s)
# #         result["Confidence"].append(c)
# #         result["Left"].append(l)
# #     return result


# # dataframe=pd.DataFrame(inspect(results))
# # dataframe.to_csv("results.csv", index=False)


# # from collections import defaultdict
# # import numpy as np
# # from typing import List

# # # same value and same order  below will not work.


# # class Solution:
# #     def equalPairs(self, grid: List[List[int]]) -> int:
# #         z = np.asarray(grid)
# #         n = len(z)
# #         count = 0
# #         for row in range(n):
# #             for col in range(n):
# #                 if all(z[row] == z[:, col]):
# #                     count += 1
# #         return count

# from typing import List


# class Solution:
#     def equalPairs(self, grid: List[List[int]]) -> int:
#         n = len(grid)
#         hash_map = {}

#         for i in range(n):
#             row = tuple(grid[i])
#             if row in hash_map:
#                 hash_map[row] += 1
#             else:
#                 hash_map[row] = 1

#         columns = zip(*grid)
#         count = 0
#         for col in columns:
#             if col in hash_map:
#                 count += hash_map[col]

#         return count


# grid = [[3, 1, 2, 2],
#         [1, 4, 4, 5],
#         [2, 4, 2, 2],
#         [2, 4, 2, 2]]

# # print(tuple(zip(*grid)))
# print(Solution().equalPairs(grid))

# class Solution:
#     def backspaceCompare(self, s: str, t: str) -> bool:
#         # fist solution will be by stack and it might not be the best solution
#         first = []
#         second = []
#         for i in range(len(s)):
#             if s[i] != "#":
#                 first.append(s[i])
#             else:
#                 if first:
#                     first.pop()

#         for i in range(len(t)):
#             if t[i] != "#":
#                 second.append(t[i])
#             else:
#                 if second:
#                     second.pop()

#         return "".join(first) == "".join(second)


# Solution().backspaceCompare("ab#c", "ad#c")

from typing import List


class Solution:
    def minOperations(self, logs: List[str]) -> int:
        files = []
        for i in logs:
            if i != "../" and i != "./":
                files.append(i)
            elif i == "./":
                continue
            else:
                # it is a file
                if files:
                    files.pop()

        return len(files)


sets = ["d1/", "d2/", "../", "d3/", "./"]

print(Solution().minOperations(sets))
