import numpy as np
from cumquat import *
import pdb
import math

class UKF:
    def __init__(self):
        self.x = np.array([1.0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        self.P = 1 * np.identity(6)
        self.Q = 8 * np.identity(6)
        self.R = 8 * np.identity(6)

    def x_omega(self):
        return self.x[4:]
    
    def x_quat(self):
        return self.x[:4]
    
    def cholesky(self, B):
        n = B.shape[0]
        L = np.zeros((n, n))

        for i in range(n):
            for k in range(i+1):
                t_sum = sum(L[i, j] * L[k, j] for j in range(k))

                if i == k:
                    L[i, k] = math.sqrt(B[i, i] - t_sum)
                else:
                    L[i, k] = (1.0 / L[k, k] * (B[i, k] - t_sum))
        return L

    def orientation_to_quat(self, w, dt=1.0):
        """ Convert 3D Orientation to 4D quat """
        q = np.array([1.0, 0.0, 0.0, 0.0])
        if np.linalg.norm(w) != 0:
            angle = np.linalg.norm(w) * dt
            axis = w * 1.0/np.linalg.norm(w)
            q[0] = np.cos(angle/2.)
            q[1:] = axis * np.sin(angle/2.)
        return q

    def calc_sigma_pts(self):
        n = self.P.shape[0]

        #Eq 35, 36
        S = self.cholesky(self.P + self.Q)
        W_plus = math.sqrt(2*n) * S
        W_minus = -1.0 * math.sqrt(2*n) * S
        W = np.hstack((W_plus, W_minus))
    
        #Create X 
        q = self.x_quat()
        omega = self.x_omega()
        m = W.shape[1]
        X = np.empty((7, m))
        for i in range(m):
            #Eq 34
            w_q = self.orientation_to_quat(W[:3, i]) #convert 3D to 4D
            w_omega = W[3:, i] 
            new_q = quat_mult(q, w_q)
            new_omega = omega + w_omega

            X[0:4, i] = new_q
            X[4:, i] = new_omega
        return X
    
    def quat_to_rotation_vector(self, q):
        angle = math.acos(q[0]) * 2.0
        axis = q[1:] / math.sin(angle/2.0)
        r = axis * angle
        return r

    def get_q_omega(self, y):
        return y[:4], y[4:]

    def compute_covariance(self, Y, y_hat):
        m = Y.shape[1]
        n = 6
        W_prime = np.zeros((n, m))
        q_yhat = y_hat[:4]
        omega_yhat = y_hat[4:]

        for i in range(m):
            q_y = Y[:4, i]
            omega_y = Y[4:, i]
            omega_wprime = omega_y - omega_yhat

            #Eq 67 #FLIPPED SIGNS?
            r_wprime = self.quat_to_rotation_vector(quat_mult(q_y, quat_inv(q_yhat)))
            W_prime[0:3, i] = r_wprime #4D back to 3D
            W_prime[3:, i] = omega_wprime

        P = (W_prime @ W_prime.T) * 1./(2.*n)
        return P, W_prime

    def process_update(self, dt):
        X = self.calc_sigma_pts()
        m = X.shape[1]

        #1) Construct Y by stepping X forward
        Y = np.zeros((X.shape), dtype=np.float64)

        #a) Update Quaternion portion of each Column in X (X->Y)
        for i in range(m):
            q = X[0:4, i]
            omega = X[4:, i]
            d_q = self.orientation_to_quat(X[4:, i], dt=dt)

            Y[0:4, i] = quat_mult(q, d_q)
            Y[4:, i] = omega


        #2) Retrieve Mean and Covariance from Y
        y_hat = np.zeros(7, dtype=np.float64)
        mean_q = quat_avg(Y[:4, :])
        mean_omega = np.mean(Y[4:, :], axis=1)

        y_hat[:4] = mean_q
        y_hat[4:] = mean_omega
        P_y, W_prime = self.compute_covariance(Y, y_hat)

        #3) Update the state and covariance matrices
        self.x = y_hat
        self.P = P_y

        return W_prime, Y
    
    def _H2(self, q_y):
        g = np.array([0, 0, 0, 1])
        inv_q_y = quat_inv(q_y)
        g_prime = quat_mult(quat_mult(inv_q_y, g), q_y) #Eq 27
        return g_prime[1:]

    def _H1(self, omega_y):
        return omega_y

    def measurement_update(self, imu_data, W_prime, Y):
        """ imu_data: [Ax, Ay, Az, Wx, Wy, Wz]
            W_prime: 6 x 2n Matrix (See Diagram)
            Y: 7 x 2n Matrix (See Diagram)
        """
        n, m = W_prime.shape
        #1) Project Y -> Z => P(Z|X=x)
        #This is what we 'expect' sensor readings to look like given our state est
        Z = np.zeros((6, m))
        for i in range(m):
            q_y, omega_y = self.get_q_omega(Y[:, i])


            #a) Apply H1 & H2 (Eq 24, 29)
            z_rot = self._H1(omega_y)
            z_acc = self._H2(q_y)

            Z[:3, i] = z_acc
            Z[3:, i] = z_rot


        #2) Calculate mean and covariance of Z, P_zz, P_vv, P_xz
        mean_z = np.mean(Z, axis=1)
        Z_minus_mean = (Z.T - mean_z).T
        P_zz = (1./(2.*n)) * (Z_minus_mean @ Z_minus_mean.T)
        P_vv = P_zz + self.R
        P_xz = (1./(2.*n)) * (W_prime @ Z_minus_mean.T)


        #3) Kalman Gain and State Update Equations
        K = P_xz @ np.linalg.inv(P_vv)
        v = imu_data - mean_z
        imu_update_6d = K @ v
        imu_update_7d = np.zeros(7)
        imu_update_7d[0:4] = self.orientation_to_quat(imu_update_6d[:3])
        imu_update_7d[4:] = imu_update_6d[3:]

        #a) Compose new quaternion
        x_pos = np.zeros(7)
        x_pos[:4] = quat_mult(self.x_quat(), imu_update_7d[:4])
        x_pos[4:] = self.x_omega() + imu_update_7d[4:]
        P = self.P - (K @ P_vv @ K.T)

        self.x = x_pos
        self.P = P

# def unit_test():
#     ukf = UKF()

#     print("------Testing Cholesky-----------")
#     A = np.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]])
#     L_est = ukf.cholesky(A)
#     L_true = np.array([[2.449489742783178, 0.0, 0.0, 0.0],
#  [1.2247448713915892, 2.1213203435596424, 0.0, 0.0],
#  [1.6329931618554523, 1.414213562373095, 2.309401076758503, 0.0],
#  [3.2659863237109046,
#   -1.4142135623730956,
#   1.5877132402714704,
#   3.1324910215354165]])

#     print(L_est)
#     print(L_true)
#     print(f"Error:{np.linalg.norm(L_true - L_est)}")
#     print("---"*12)

#     print("-------Testing Process Update ---------")
#     W_prime, Y = ukf.process_update(0.1)
#     print("MEAN:", ukf.x, "\nSUM:", np.sum(ukf.x))
#     print("COV:\n", ukf.P, "\nTRACE:", np.trace(ukf.P), "\nSUM:", np.sum(ukf.P))

#     #SAME SO FAR!

#     print("-------Testing Measurement Update ---------")
#     test_imu_data = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
#     ukf.measurement_update(test_imu_data, W_prime, Y)
#     print("MEAN:", ukf.x, "\nSUM:", np.sum(ukf.x))
#     print("COV:\n", ukf.P, "\nTRACE:", np.trace(ukf.P))

    
#     print("------Sanity Testing Several Process + Measurement Updates")
#     ukfnew = UKF()
#     test_imu_data = np.random.rand(6)
#     for i in range(10):
#         # W_prime, Y = ukfnew.process_update(0.01)
#         # ukfnew.measurement_update(test_imu_data, W_prime, Y)
#         print("i:", i, ", TRACE COV:", np.trace(ukfnew.P))
# unit_test()