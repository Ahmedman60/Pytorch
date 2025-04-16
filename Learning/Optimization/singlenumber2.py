class Solution:
    def singleNumber(self, nums):
        seen_once = 0
        seen_twice = 0

        for n in nums:
            # if seen_once =0  not seen  add it to seen once
            if seen_once & (1 << n) == 0 and seen_twice & (1 << n) == 0:
                seen_once |= n
            else:
                # else mean you are seeing it again  remove it from seen once
                seen_once &= ~(1 << n)
                # add it to see twice so you don't add it again
                seen_twice |= n

        return seen_once


print(Solution().singleNumber([2, 2, 3, 2]))


class Solution:
    def duplicateNumbersXOR(self, nums) -> int:
        mask = 0
        result = 0
        for i in nums:
            # if number exist in mask
            if mask & (1 << i):
                result ^= i
            else:
                mask |= (1 << i)

        return result
