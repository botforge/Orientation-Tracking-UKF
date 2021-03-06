from scipy import io
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb
import time
import os
import cumquat
from jigglypuff import UKF

def load_imu_data(data_num):
    """
    Returns:
    - imu_raw: {"vals": 6xn np.array [Ax Ay Az Wz Wx Wy], "ts": timestamps}
    - vicon_raw: {"rots": 3x3xn np.array Rotation Matrix, "ts": timestamps}
    """
    imu_filename = os.path.join(os.path.dirname(__file__), "imu/imuRaw" + str(data_num) + ".mat")
    imu_raw = io.loadmat(imu_filename)

    return imu_raw

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
def plot_vicon(vicon_raw, ax):
    rolls, pitches, yaws = vicon_to_euler(vicon_raw)
    ts = vicon_raw.get("ts")
    ax[0].plot(np.squeeze(ts), rolls, label="roll")
    ax[1].plot(np.squeeze(ts), pitches, label="pitch")
    ax[2].plot(np.squeeze(ts), yaws, label="yaw")

def process_imu(imu_raw, rpy=False, datanum=1):
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
    if datanum == 3:
        bias_accel = accel[0] - zero_g
    accel = (accel - bias_accel)

     #b) Gyroscope
    sens_gyro = 3.33
    scale_gyro = 3300/1023/sens_gyro
    gyro = gyro * scale_gyro
    # bias_gyro = gyro[0]
    bias_gyro = np.mean(gyro[0:300], axis=0)
    if datanum == 3:
        bias_gyro = np.mean(gyro[0:250], axis=0)
    gyro = (gyro - bias_gyro) * (math.pi/180.0)

    return accel, gyro, ts


def euler_angles(q):
          r = math.atan2(2*(q[0]*q[1]+q[2]*q[3]), \
                  1 - 2*(q[1]**2 + q[2]**2))
          p = math.asin(2*(q[0]*q[2] - q[3]*q[1]))
          y = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), \
                  1 - 2*(q[2]**2 + q[3]**2))
          return np.array([r, p, y])
def estimate_rot(data_num=1, plot=False, use_vicon = False):
    if use_vicon:
        imu_raw, vicon_raw = load_data(data_num)
    else:
        imu_raw = load_imu_data(data_num)
    if plot:
        fig, axes = plt.subplots(nrows=3, ncols=1)
        plot_vicon(vicon_raw, axes)
    accel, gyro, ts = process_imu(imu_raw, rpy=False, datanum = data_num)
    imu_data = np.hstack((accel, gyro))
    data_len = imu_data.shape[0]
    
    ukf = UKF()
    if data_num == 2:
        ukf.P = 3000 * np.identity(6)
        ukf.Q = 100 * np.identity(6)
        ukf.R = 100 * np.identity(6)
    elif data_num == 1:
        ukf.P = 2500 * np.identity(6)
        ukf.Q = 100 * np.identity(6)
        ukf.R = 100 * np.identity(6)
    elif data_num == 3:
        ukf.P = 1 * np.identity(6)
        ukf.Q = 8 * np.identity(6)
        ukf.R = 8 * np.identity(6)
    
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

    #3) Convert [Rx, Ry, Rz] into roll, pitch, yaw (roll & pitch from accel are swapped w.r.t vicon)
    #a) Accelerometer
    # ukf_pitch = np.arctan2(-accel[:, 0], accel[:, 2])[:-1]
    # ukf_rolls = np.arctan2(accel[:, 1], (np.sqrt(np.square(accel[:, 0]) + np.square(accel[:, 2]))))[:-1]

    if plot:
        axes[0].plot(np.squeeze(ts)[:-1], ukf_rolls, label="ukf_rolls")
        axes[0].legend()
        axes[1].plot(np.squeeze(ts)[:-1], ukf_pitch, label="ukf_pitch")
        axes[1].legend()
        axes[2].plot(np.squeeze(ts)[:-1], ukf_yaw, label="ukf_yaw")
        axes[2].legend()
        plt.show()

    return np.array(ukf_rolls), np.array(ukf_pitch), np.array(ukf_yaw)

data_num=1
start = time.time()
estimate_rot(data_num, plot=True, use_vicon=True)
end = time.time()
print("done, ", end-start)