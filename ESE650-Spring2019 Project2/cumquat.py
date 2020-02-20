import numpy as np
import math

def euler_angles(q):
        r = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), \
                1 - 2*(q[1]**2 + q[2]**2))
        p = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
        y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), \
                1 - 2*(q[2]**2 + q[3]**2))
        return np.array([r, p, y])

def from_rotm(R):
    theta = math.acos((np.trace(R)-1)/2)
    omega_hat = (R - np.transpose(R))/(2*math.sin(theta))
    omega = np.array([omega_hat[2,1], -omega_hat[2,0], omega_hat[1,0]])
    q[0] = math.cos(theta/2)
    q[1:4] = omega*math.sin(theta/2)
    normalize(q)
    return q

def normalize(q):
    q = 1/np.linalg.norm(q) * q

def quat_avg(Q):
    """ Q: 4 x n matrix """ 
    w, v = np.linalg.eig(Q @ Q.T)
    v[:, np.argmax(w)] = 