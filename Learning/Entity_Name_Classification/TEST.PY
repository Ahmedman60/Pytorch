
# # FIBO iterative
# def fibo(n):
#     a, b = 0, 1
#     c = 0
#     for i in range(n):
#         a, b = b, a+b
#         # a = b
#         # b = c
#         # c = a+b
#     return a


# # print(fibo(7))  # 5

# # LCS algorithm


# # def lcs(m, n):
# #     if m == 0 or n == 0:
# #         return 0
# #     elif m == n:
# #         return 1
# #     else:
# #         if m > n:
# #             return lcs(m-1, n)
# #         else:
# #             return lcs(m, n-1)

# # Memorization , in lecturecture they take Tabulazation.
# def lcs(s1, s2, m, n, memo):
#     if m == 0 or n == 0:
#         return 0
#     if (m, n) in memo:
#         return memo[(m, n)]

#     if s1[m - 1] == s2[n - 1]:
#         memo[(m, n)] = 1 + lcs(s1, s2, m - 1, n - 1, memo)
#     else:
#         memo[(m, n)] = max(lcs(s1, s2, m, n - 1, memo),
#                            lcs(s1, s2, m - 1, n, memo))
#     return memo[(m, n)]


# if __name__ == "__main__":
#     s1 = "AGGTAB"
#     s2 = "GXTXAYB"

#     m = len(s1)
#     n = len(s2)
#     memo = {}  # Initialize memoization dictionary
#     print(lcs(s1, s2, m, n, memo))

#     print([[-1 for _ in range(n + 1)] for _ in range(m + 1)])


# # s1 = "AGGTAB"
# # s2 = "GXTXAYB"
# # n = len(s1)
# # m = len(s2)
# memory = [[-1 for j in range(8)] for i in range(9)]

# # geting dimension of memory
# print(len(memory), len(memory[0]))

# print(memory)
