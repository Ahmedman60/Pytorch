
# FIBO iterative
def fibo(n):
    a, b = 0, 1
    c = 0
    for i in range(n):
        a, b = b, a+b
        # a = b
        # b = c
        # c = a+b
    return a


print(fibo(7))  # 5
