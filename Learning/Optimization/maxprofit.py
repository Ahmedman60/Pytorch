class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        max_pofit = 0
        min_price = float('inf')
        for i in range(len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            else:
                profit = i - prices[i]
                max_pofit = max(max_pofit, profit)

        return max_pofit


# Test case
# Expected output: 5 (buy at 1 and sell at 6)
print(Solution().maxProfit([7, 1, 5, 3, 6, 4]))
# Expected output: 0 (no transaction is done)
print(Solution().maxProfit([3, 2, 6, 1, 5]))
