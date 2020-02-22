import numpy as np
import math

def euler_angles(q):
    r = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), \
	1 - 2*(q[1]**2 + q[2]**2))
    p = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
    y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), \
    1 - 2*(q[2]**2 + q[3]**2))
    return np.array([r, p, y])

    
def normalize(q):
    q = q * (1/np.linalg.norm(q))
    return q

def from_rotm(R, q):
    theta = math.acos((np.trace(R)-1)/2)
    omega_hat = (R - np.transpose(R))/(2*math.sin(theta))
    omega = np.array([omega_hat[2,1], -omega_hat[2,0], omega_hat[1,0]])
    q[0] = math.cos(theta/2)
    q[1:4] = omega*math.sin(theta/2)
    normalize(q)
    return q

def quat_mult(q0, q1):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1

    q =  np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                        x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)
    return q

def quat_inv(q):
    inv_q = q*-1.0
    inv_q[0] *= -1.0
    inv_q /= np.linalg.norm(inv_q) ** 2
    return inv_q

def quat_avg(Q):
    """ Q: 4 x n matrix """ 
    w, v = np.linalg.eig(Q @ Q.T)
    q = v[:, np.argmax(w)]
    return normalize(q)

def build_quat(w, x, y, z):
    return np.array([w, x, y, z])

#TESTING#
def oracle_quatmul(quaternion0, quaternion1):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                        x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0], dtype=np.float64)

def unit_tests():
    q1 = build_quat(1, 0, 1, 0)
    q2 = build_quat(1, 0, 0.5, 1)

    print("-----------Testing Quaternion Multiplication..")
    q3_est = quat_mult(q1, q2)
    q3_true = oracle_quatmul(q1, q2)
    print(f"Error:{np.linalg.norm(q3_true - q3_est)}, Diff:{q3_true - q3_est}")


    q1 = np.array([0.1234567, 0.2345678, 0.3456789, 0.45678901])
    q2 = np.array([0.2345678, 0.3456789, 0.45678901, 0.1234567])
    prod = quat_mult(normalize(q1), normalize(q2))
    print(prod)

    print("-----------Testing Quaternion Averaging")
    from scipy.spatial.transform import Rotation as R

    r = R.from_quat([[0, 0, np.sin(np.pi/4), np.cos(np.pi/4)],
                    [0, 0, np.sin(np.pi / 2), np.cos(np.pi / 2)],
                    [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)],
                    [0, 0, np.sin(np.pi / 16), np.cos(np.pi / 16)]])

    meanr = r.mean()

    Q = np.array([[0, 0, np.sin(np.pi/4), np.cos(np.pi/4)],
                    [0, 0, np.sin(np.pi / 2), np.cos(np.pi / 2)],
                    [0, 0, np.sin(np.pi / 8), np.cos(np.pi / 8)],
                    [0, 0, np.sin(np.pi / 16), np.cos(np.pi / 16)]]).T

    qavg_true = meanr.as_quat()
    qavg_est = quat_avg(Q)
    print(f"qavg_true:{qavg_true}")
    print(f"qavg_est:{qavg_est}")
    print(f"Error:{np.linalg.norm(qavg_true - qavg_est)}")


    print("-----------Testing Inverse Quaternion")
    q = np.array([1, 0, 1, 0])
    print(quat_inv(q))
    print(quat_mult(q, quat_inv(q)))
    
unit_tests()