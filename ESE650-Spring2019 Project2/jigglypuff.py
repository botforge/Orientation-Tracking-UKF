import numpy as np
from math import sqrt

class UKF:
    def __init__(self):
        self.x = np.zeros((7,))
        self.P = np.zeros((6, 6))
        self.k = 0

    def cholesky(self, B):
        n = B.shape[0]
        L = np.zeros((n, n))

        for i in range(n):
            for k in range(i+1):
                t_sum = sum(L[i, j] * L[k, j] for j in range(k))

                if i == k:
                    L[i, k] = sqrt(B[i, i] - t_sum)
                else:
                    L[i, k] = (1.0 / L[k, k] * (B[i, k] - t_sum))
        return L

    def calc_sigma_pts(self, )

    

def unit_test():
    ukf = UKF()

    print("------Testing Cholesky-----------")
    A = np.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]])
    L_est = ukf.cholesky(A)
    L_true = np.array([[2.449489742783178, 0.0, 0.0, 0.0],
 [1.2247448713915892, 2.1213203435596424, 0.0, 0.0],
 [1.6329931618554523, 1.414213562373095, 2.309401076758503, 0.0],
 [3.2659863237109046,
  -1.4142135623730956,
  1.5877132402714704,
  3.1324910215354165]])

    print(L_est)
    print(L_true)
    print(f"Error:{np.linalg.norm(L_true - L_est)}")
    print("---"*12)

unit_test()