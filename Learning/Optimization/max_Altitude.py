# # class Solution(object):
# #     def largestAltitude(self, gain):
# #         """
# #         :type gain: List[int]
# #         :rtype: int
# #         """
# #         max_altitude = 0
# #         current = 0
# #         for i in gain:
# #             current += i
# #             max_altitude = max(max_altitude, current)
# #         return max_altitude


# # class Solution(object):
# #     def prefix(self, gain):
# #         """
# #         :type gain: List[int]
# #         :rtype: int
# #         """
# #         if len(gain) == 1:
# #             return max(gain[0], 0)
# #         max_altitude = 0
# #         for i in range(1, len(gain)):
# #             gain[i] += gain[i-1]  # Compute cumulative altitude
# #             max_altitude = max(max_altitude, gain[i])
# #         return max_altitude


# # class Solution(object):
# #     def prefix(self, gain):
# #         """
# #         :type gain: List[int]
# #         :rtype: int
# #         """
# #         max_altitude = 0
# #         current_altitude = 0  # Explicitly track the altitude
# #         for i in range(1, len(gain)):
# #             gain[i] += gain[i-1]  # Compute cumulative altitude
# #             current_altitude = gain[i]
# #             max_altitude = max(max_altitude, current_altitude)
# #         return max_altitude, gain


# # print(Solution().prefix([5, -10, 3, 2]))


# # def prefix_sum(arr):
# #     """
# #     :type arr: List[int]
# #     :rtype: List[int]
# #     """
# #     if not arr:
# #         return []

# #     prefix = [arr[0]]  # First element remains the same
# #     for i in range(1, len(arr)):
# #         prefix.append(prefix[-1] + arr[i])  # Add current element to last sum

# #     return prefix


# # print(prefix_sum([5, -10, 3, 2]))  # Example usage of prefix_sum function


# # incase i need to modify inplace but this not the best.
# class Solution(object):
#     def largestAltitude(self, gain):
#         """
#         :type gain: List[int]
#         :rtype: int
#         """
#         if len(gain) == 1:
#             return max(gain[0], 0)
#         max_altitude = 0
#         current_altitude = gain[0]  # Explicitly track the altitude
#         for i in range(1, len(gain)):
#             gain[i] += gain[i-1]
#             max_altitude = max(max_altitude, current_altitude)
#             current_altitude = gain[i]
#         return max_altitude


# print(Solution().largestAltitude([5, -10, 3, 2]))

# Best solution


class Solution(object):
    def largestAltitude(self, gain):
        """
        :type gain: List[int]
        :rtype: int
        """
        if len(gain) == 1:
            return max(gain[0], 0)

        max_altitude = 0
        current_altitude = 0

        for i in range(len(gain)):
            current_altitude += gain[i]
            max_altitude = max(max_altitude, current_altitude)

        return max_altitude
