

import csv
import numpy as np 
import time 
import matplotlib.pyplot as plt
from numpy.linalg import inv


data, data_new_pos, velocity, std_vel, std_pos, corr_coef = [], [], [], [], [], []

def read_data():
	with open('RBE500-F17-100ms-Constant-Vel.csv', 'r') as datafile:
		data1 = csv.reader(datafile)
		for d in data1:
			data.append(int(d[0]))

def plot_data():
	# print (data)
	plt.figure(1)
	plt.plot(data, 'g*-', label="Raw Data")
	plt.plot(data_new_pos, 'r-', label="Kalman Filter Processed data" )
	plt.xlabel('Time elapsed, in intervals of 1 unit = 100 ms ')
	plt.ylabel('Distance between the sensor and wall, in intervals of 1 unit = 1 centimeter')
	plt.title('Part A - Distance from the wall versus time plot for with and without Kalman Filtering')
	plt.legend(loc=1, borderaxespad=0.)
	plt.grid()

	plt.figure(2)
	plt.plot(velocity)
	plt.xlabel('Time elapsed, in intervals of 1 unit = 100 ms ')
	plt.ylabel('Velocity of the sensor, 1 unit = 1 cm/s')
	plt.title('Part A - Velocity of the sensor as updated after Kalman Filtering')
	plt.grid()

	plt.figure(3)
	plt.plot(std_pos, 'r-', label ="Standard Deviation in position")
	plt.plot(std_vel, 'g-', label="Standard Deviation in velocity")
	plt.plot(corr_coef, 'b--', label="Correlation Coefficient")
	plt.xlabel('Time elapsed, in intervals of 1 unit = 100 ms ')
	plt.ylabel(' Standard deviations for velocity and position and Value of Correlation coefficient')
	plt.title('Part A - Standard deviations for velocity and position and Value of Correlation coefficient vs time')
	plt.legend(loc=1, borderaxespad=0.)
	plt.grid()
	plt.show()


def kalman_filter(del_t,r, debug):

	X_o = np.array([[2530], [-10]])									# Initial State matrix containing the initial position and velocity
	P_o = np.array([[100, 0],[0, 100]])								# Prediction covariance matrix
	X_current_updated = X_o
	P_current_updated = P_o

	for i in range(0, len(data)):

		# Kalman Gain is a way to give weight factor to the estimation as well as to the measurement depending upon the reliability of the two
		
		# Making the last updated values as the initial values to start the new iteration with 
		X_previous = X_current_updated
		P_previous = P_current_updated

		# State transition matrix A 	
		A = np.array([[1, del_t],[0, 1]])

		# Process plant noise covariance matrix R	 
		R  = np.matmul(np.array([[del_t], [1]]), np.array([[r*del_t**2, r*del_t]]))										

		# Calculate current prediction of State matrix and covariance matrix
		X_current_predicted = np.matmul(A,X_previous)
		# print ("X current estimation", X_current_predicted)
		P_current_predicted = np.add(np.matmul(np.matmul(A, P_previous), np.transpose(A)), R) 

		# Measurement covariance matrix - Lidar measurement noise variance = 10 
		Q = np.array([10])																
		
		C = np.array([1,0])

		# Transformation matrix H
		H = np.array([[1,0]])

		# Calculate Kalman Gain
		K_gain_numerator = 	np.matmul(P_current_predicted, np.transpose(H))
		K_gain_denominator = np.add(np.matmul(np.matmul(H, P_current_predicted), np.transpose(H)), Q)   # P_current_predicted, R)
		K_gain = K_gain_numerator/K_gain_denominator
		# print ("K gain num and deno", K_gain_numerator, K_gain_denominator)

		# Calculate the final value of the state variables and the covariance matrix
		X_current_updated = np.add(X_current_predicted,  K_gain*(np.subtract(data[i],np.matmul(H, X_current_predicted))))
		P_current_updated = np.matmul(np.subtract(np.array([[1,0], [0,1]]), np.matmul(K_gain, H)), P_current_predicted)

		# Extracting the positions from the final predictions
		data_new_pos.append(X_current_updated[0][0])
		# Extracting the velocities fromt the final predictions
		velocity.append(X_current_updated[1][0])

	
		# Standard deviation changing with time for position and velocity
		std_pos.append(np.sqrt(P_current_updated[0][0])) 
		std_vel.append(np.sqrt(P_current_updated[1][1]))
		# print ("Std_pos", std_pos[i])
		# print ("Std velo", std_vel[i])

		# Correlation coefficient for velocity and position
		corr_coef.append(P_current_updated[0][1]/(std_pos[i]*std_vel[i]))


		# If in debug mode, print the various values
		if debug:

			print ("Initial values", X_o )
			print ("Initial covariance matrix", P_o)
			print ("State transition matrix A", A)
			print ("Process plant noise covariance matrix R", R)
			print ("Current prediction of X and P", X_current_predicted, P_current_predicted)
			print ("Lidar measurement covariance matrix Q", Q)
			print ("Input measurement transition matrix C", C)
			print ("Transformation matrix H", H)
			print ("Kalman Gain", K_gain)
			print ("Input value", data[i])
			print ("Finally updated values of state and covariance matrix", X_current_updated, P_current_updated)
			print (" --------------------------------------------------------------------------------------")
			print (" \n \n ")

read_data()
kalman_filter(0.1, 100, True) 							# Use False to skip data printing
plot_data()
