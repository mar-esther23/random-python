
def fib():
    f1, f2 = 0,1
    while True:
        yield f1 + f2
        f1, f2 = f2, f1+f2

def base_convert(n, base):
    c = 0
    while c < base ** n :
        s = [0 for i in range(n)]
        x, i = c, 1
        while x > 0: 
            # print x, i, ' ',
            s[-i] = x%base
            x = x/base
            i += 1
        c+=1
        yield s

f = fib()
for i in range(10):
    print i, next(f)

for s in base_convert(4, 3):
    print s