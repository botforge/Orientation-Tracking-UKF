from scipy import io
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb
import os
import cumquat
from jigglypuff import UKF

def load_data(data_num):
    """
    Returns:
    - imu_raw: {"vals": 6xn np.array [Ax Ay Az Wz Wx Wy], "ts": timestamps}
    - vicon_raw: {"rots": 3x3xn np.array Rotation Matrix, "ts": timestamps}
    """
    imu_filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat")
    imu_raw = io.loadmat(imu_filename)

    vicon_filename = os.path.join(os.path.dirname(__file__), "vicon/viconRot" + str(data_num) + ".mat")
    vicon_raw = io.loadmat(vicon_filename)
    return imu_raw, vicon_raw

def R_to_euler(R):
    "Convert 3x3 R matrix into roll, pitch, yaw"
    yaw = math.atan(R[1, 0]/R[0, 0])
    pitch = math.atan(-1.0 * R[2,0] / np.sqrt(R[2,1]**2 + R[2,2]**2))
    roll = math.atan(R[2, 1]/R[2, 2])
    return roll, pitch, yaw

def vicon_to_euler(vicon_raw):
    "Get RPY array for each 3x3 R matrix in vicon_raw"
    rots, ts = vicon_raw.get("rots"), vicon_raw.get("ts")
    ts = np.squeeze(ts)
    rolls = []
    pitches = []
    yaws = []
    for i in range(rots.shape[-1]):
        R = rots[:, :, i]
        roll, pitch, yaw = R_to_euler(R)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)
    return rolls, pitches, yaws

def plot_vicon(vicon_raw):
    rolls, pitches, yaws = vicon_to_euler(vicon_raw)
    ts = vicon_raw.get("ts")
    plt.plot(np.squeeze(ts), rolls, label="roll")
    plt.plot(np.squeeze(ts), pitches, label="pitch")
    plt.plot(np.squeeze(ts), yaws, label="yaw")

def process_imu(imu_raw, rpy=False):
    """
    """
    vals = imu_raw.get("vals")
    ts = np.squeeze(imu_raw.get("ts"))
    zero_g = np.array([0, 0, +1])

    #1) Extract accel and gyro from measurements & transform
    #a) Transform Accel according to note@183
    tf_accel = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    accel = vals[:3, :]
    accel = (tf_accel @ accel).T

    #b) Gyro according to note@183
    # tf_gyro = np.array([
    #     [0, 0, 1],
    #     [1, 0, 0],
    #     [0, 1, 0]
    # ])
    tf_gyro = np.array(
        [[0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]]
    )
    gyro = vals[3:, :]
    gyro = (tf_gyro @ gyro).T
    
    #2) Convert ADC -> mV -> delta_mV -> [Rx, Ry, Rz] (force values in each direction)
    #a) Accelerometer
    sens_accel = 330
    scale_accel = 3300/1023/sens_accel #IMU Ref sheet
    accel = accel * scale_accel
    bias_accel = np.mean(accel[0:10], axis=0) - zero_g 
    accel = (accel - bias_accel)

     #b) Gyroscope
    sens_gyro = 3.33
    scale_gyro = 3300/1023/sens_gyro
    gyro = gyro * scale_gyro
    bias_gyro = np.mean(gyro[0:10], axis=0)
    gyro = (gyro - bias_gyro) * (math.pi/180.0)

    # if rpy:
    #     #3) Convert [Rx, Ry, Rz] into roll, pitch, yaw (roll & pitch from accel are swapped w.r.t vicon)
    #     #a) Accelerometer
    #     accel_pitch = np.arctan2(-accel[:, 0], accel[:, 2])
    #     accel_roll = np.arctan2(accel[:, 1], (np.sqrt(np.square(accel[:, 0]) + np.square(accel[:, 2]))))
    #     plt.plot(ts, accel_pitch, label="accel_pitch")
    #     plt.plot(ts, accel_roll, label="accel_roll")

    #     #b) Gyroscope
    #     omega = gyro
    
    return np.hstack((accel, gyro)), ts


# def quat_euler_angles(q):
#     r = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), \
# 	1 - 2*(q[1]**2 + q[2]**2))
#     p = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
#     y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), \
#     1 - 2*(q[2]**2 + q[3]**2))
#     return r, p, y

def euler_angles(q):
          r = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), \
                  1 - 2*(q[1]**2 + q[2]**2))
          p = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
          y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), \
                  1 - 2*(q[2]**2 + q[3]**2))
          return np.array([r, p, y])

def estimate_rot(data_num=1, plot=False, use_dset = False):
    if use_dset:
        imu_raw, vicon_raw = load_data(data_num)
    if plot:
        plot_vicon(vicon_raw)
    imu_data, ts = process_imu(imu_raw, rpy=False)
    data_len = imu_data.shape[0]

    ukf = UKF()
    ukf_rolls = []
    ukf_pitch = []
    ukf_yaw = []
    for i in range(1, data_len):
        dt = ts[i] - ts[i-1]
        W_prime, Y = ukf.process_update(dt)
        ukf.measurement_update(imu_data[i, :], W_prime, Y)
        r, p, y = euler_angles(ukf.x[:4])
        ukf_pitch.append(p)
        ukf_rolls.append(r)
        ukf_yaw.append(y)

    if plot:
        plt.plot(np.squeeze(ts)[:-1], ukf_rolls, label="ukf_rolls")
        plt.legend()
        plt.show()
    
    return np.array(ukf_rolls), np.array(ukf_pitch), np.array(ukf_yaw)

# data_num=1
# estimate_rot(data_num, plot=True)