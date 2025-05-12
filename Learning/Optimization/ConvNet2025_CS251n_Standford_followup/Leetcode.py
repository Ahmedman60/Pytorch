# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:

        # if one element don't do below code return None
        if not head or head.next is None:
            return None
        # basic and bad solution
        lenght = 0
        currentNode = head
        while currentNode:
            lenght += 1
            currentNode = currentNode.next

        currentNode = head
        pervious = None
        i = 1
        mid = lenght//2
        while i <= mid:
            pervious = currentNode
            currentNode = currentNode.next
            i += 1
        # remove the current Node
        pervious.next = currentNode.next
        currentNode.next = None

        return head


class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Slow and fast pointer
        if not head or head.next is None:
            return None
        slow = fast = head
        pervious = None
        while fast and fast.next:
            # while fast not out of bound , move slow step and fast 2 steps
            pervious = slow
            slow = slow.next
            fast = fast.next.next

        # remove the slow
        pervious.next = slow.next
        slow.next = None

        return head


# testing the code
if __name__ == "__main__":
    # Create a sample linked list: 1 -> 2 -> 3 -> 4 -> 5
    head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
    head = ListNode(5, ListNode(4))
    # Call the deleteMiddle function
    solution = Solution()
    result = solution.deleteMiddle(head)

    # Print the modified linked list
    while result:
        print(result.val, end=" ")
        result = result.next


# # Definition for singly-linked list.
# # class ListNode:
# #     def __init__(self, val=0, next=None):
# #         self.val = val
# #         self.next = next
# class Solution:
#     def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         slow = fast = head
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next

#         return slow
