class Solution(object):
    def findDifference(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[List[int]]
        """
        num_set1 = set(nums1)
        num_set2 = set(nums2)
        return [list(num_set1.difference(num_set2)), list(num_set2.difference(num_set1))]


nums1 = [1, 2, 3]
nums2 = [2, 4, 6]

print(Solution().findDifference(nums1, nums2))  # Output: [[1, 3], [4, 6]]
