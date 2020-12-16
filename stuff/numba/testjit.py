import numpy as np
import time
from numba import jit


trials = 10

N = 200

tic = time.time()
for trial in range(trials):
    x = np.zeros((N, N, N))
    for j in range(N):
        for k in range(N):
            for l in range(N):
                x[j, k, l] = j + k + l

toc = time.time()
print("Time, Using Interpreter with Loops:")
print(toc - tic)
print("")



tic = time.time()
for trial in range(trials):
    @jit(nopython=True)
    def SetXcc():
        x = np.zeros((N, N, N))
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    x[j, k, l] = j + k + l

                return x


    y = SetXcc()

toc = time.time()
print("Time Spent using loops .. with Numba JIT, with all compilations:")
t = toc - tic
print(t)
print("")

tic = time.time()


@jit(nopython=True)
def SetXc():
    x = np.zeros((N, N, N))
    for j in range(N):
        for k in range(N):
            for l in range(N):
                x[j, k, l] = j + k + l

    return x


for trial in range(trials):
    y = SetXc()

toc = time.time()
print("Time Spent using loops .. with Numba JIT, with 1 compilation:")
t = toc - tic
print(t)
print("")


@jit(nopython=True)
def SetX():
    x = np.zeros((N, N, N))
    for j in range(N):
        for k in range(N):
            for l in range(N):
                x[j, k, l] = j + k + l

    return x


y = SetX()
tic = time.time()
for trial in range(trials):
    y = SetX()

toc = time.time()
print("Time Spent using loops .. with Numba JIT, precompiled:")
t = toc - tic
print(t)
print("")

tic = time.time();
for trial in range(trials):
    d1 = np.arange(0, N, 1, dtype=np.float64)
    d1.shape = [N, 1, 1]
    d2 = d1.copy()
    d2.shape = [1, N, 1]
    d3 = d1.copy()
    d3.shape = [1, 1, N]
    y = d1 + (d2 + d3)
toc = time.time()

print("Time using vectorized implementation, no setup:")
print(toc - tic)