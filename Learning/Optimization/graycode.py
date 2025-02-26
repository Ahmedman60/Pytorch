

def solution(n):

    if n == 1:
        return [0, 1]
    else:
        result = ["0", "1"]
        while n > 1:
            solution = []
            for i in result:
                solution.append("0" + i)
            for i in result:
                solution.append("1" + i)
            result = solution
            n -= 1

        return [int(i, 2) for i in solution]


print(solution(3))

# Graycoded solution


def graycode(n):
    return [i ^ (i >> 1) for i in range(2**n)]


print(graycode(3))


def fun(n):
    while n > 1:
        print(n)
        if n % 2 == 0:
            n = n//2
        else:
            n = n*3+1

    print(1)


fun(3)
