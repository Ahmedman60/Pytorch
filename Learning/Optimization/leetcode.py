class Solution(object):
    def uniqueOccurrences(self, arr):
        """
        :type arr: List[int]
        :rtype: bool
        """
        from collections import defaultdict
        hash_maps = defaultdict(int)
        for i in arr:
            hash_maps[i] += 1

        return len(hash_maps.values()) == len(set(hash_maps.values()))


sol = Solution()
arr = [1, 2, 2, 1, 1, 3]
print(sol.uniqueOccurrences(arr))
