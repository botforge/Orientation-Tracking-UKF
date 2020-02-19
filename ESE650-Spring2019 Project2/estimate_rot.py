from scipy import io
import numpy as np
import math
import matplotlib.pyplot as plt
import pdb
import os

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

def R_to_rpy(R):
	"Convert 3x3 R matrix into roll, pitch, yaw"
	yaw = math.atan(R[1, 0]/R[0, 0])
	pitch = math.atan(-1.0 * R[2,0] / np.sqrt(R[2,1]**2 + R[2,2]**2))
	roll = math.atan(R[2, 1]/R[2, 2])
	return roll, pitch, yaw

def vicon_to_rpy(vicon_raw):
	"Get RPY array for each 3x3 R matrix in vicon_raw"
	rots, ts = vicon_raw.get("rots"), vicon_raw.get("ts")
	ts = np.squeeze(ts)
	rolls = []
	pitches = []
	yaws = []
	for i in range(rots.shape[-1]):
		R = rots[:, :, i]
		roll, pitch, yaw = R_to_rpy(R)
		rolls.append(roll)
		pitches.append(pitch)
		yaws.append(yaw)
	return rolls, pitches, yaws

def plot_vicon(vicon_raw):
	rolls, pitches, yaws = vicon_to_rpy(vicon_raw)
	ts = vicon_raw.get("ts")
	plt.plot(np.squeeze(ts), rolls, label="roll")
	plt.plot(np.squeeze(ts), pitches, label="pitch")
	plt.plot(np.squeeze(ts), yaws, label="yaw")

def process_imu(imu_raw):
	"""
	"""
	zero_g = [46.8230694, 14.56500489, 29.73607038 + 9.8]

	#1) Extract accel and gyro from measurements
	vals = imu_raw.get("vals").T
	accel = vals[:, :3]
	gyro = vals[:, 3:]
	
	#2) Convert ADC -> mV -> delta_mV
	accel = (accel * 3300) / 1023
	accel -= zero_g
	pdb.set_trace()

	#3) Convert delta_mv into roll, pitch, yaw

def estimate_rot(data_num=1):
	imu_raw, vicon_raw = load_data(data_num)
	plot_vicon(vicon_raw)
	process_imu(imu_raw)

	plt.legend()
	plt.show()

estimate_rot()