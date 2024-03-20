'''
Author: Yicheng Chen (yicheng-chen@outlook.com)
LastEditTime: 2023-03-01 23:16:14
'''

import numpy as np
import matplotlib.pyplot as plt
from polynomialgenerator import PolynomialGenerator
from header_file import *
class MinJerkPlanner():
    '''
    Minimum-jerk trajectory generator
    '''
    def __init__(self):
        self.generator = PolynomialGenerator()

    def plan(self, degrees, start_position, end_position, waypoints, durations):
        '''
        Input:
        s: degree of trajectory, 2(acc)/3(jerk)/4(snap)
        head_state: (s,D) array, where s=2(acc)/3(jerk)/4(snap), D is the dimension
        tail_state: the same as head_state
        int_wpts : (M-1,D) array, where M is the piece num of trajectory

        Note about head_state and tail_state:

        If you want a trajecory of minimum s, only the 0~(s-1) degree bounded contitions and valid.
        e.g. for minimum jerk(s=3) trajectory, we can specify
        pos(s=0), vel(s=1), and acc(s=2) in head_state and tail_state both.
        If you provide less than s degree bounded conditions, the missing ones will be set to 0.
        If you provide more than s degree bounded conditions, the extra ones will be ignored.

        head_state and tail_state are supposed to be 2-dimensional arrays.

        The function stores the coeffs in self.coeffs
        '''
        self.degrees = degrees
        self.dimensions = start_position.shape[1]
        self.number_of_splines = len(durations)
        self.n_coeffs = degrees * 2
        self.generator.set_degrees(self.n_coeffs - 1)

        input_head_shape0 = start_position.shape[0]
        input_tail_shape0 = end_position.shape[0]

        self.head_state = np.zeros((self.degrees, self.dimensions))
        self.tail_state = np.zeros((self.degrees, self.dimensions))

        for i in range(min(self.degrees, input_head_shape0)):
            self.head_state[i] = start_position[i]
        for i in range(min(self.degrees, input_tail_shape0)):
            self.tail_state[i] = end_position[i]

        self.initial_waypoints = waypoints  # 'int' for 'intermediate'
        self.durations = durations
        self.get_coefficients()

    def get_coefficients(self):
        '''
        Calculate coeffs according to int_wpts and ts
        input: int_wpts(D,M-1) and ts(M,)
        stores self.A and self.coeffs
        '''
        self.T = self.generate_time_vector()

        self.A = np.zeros((self.number_of_splines * self.n_coeffs, self.number_of_splines * self.n_coeffs))
        self.b = np.zeros((self.number_of_splines * self.n_coeffs, self.dimensions))
        self.b[0:self.degrees, :] = self.head_state
        self.b[-self.degrees:, :] = self.tail_state

        self.A[0, 0:self.n_coeffs] = self.generator.generate_polynomial(derivative=0, inverse=True)
        self.A[1, 0:self.n_coeffs] = self.generator.generate_polynomial(derivative=1, inverse=True)
        self.A[2, 0:self.n_coeffs] = self.generator.generate_polynomial(derivative=2, inverse=True)

        for i in range(self.number_of_splines - 1):
            self.set_derivatives(i)
            self.set_b(i)

        self.set_final_A()
        self.coefficients = np.linalg.solve(self.A, self.b)

    def generate_time_vector(self):
        return np.vander(self.durations, N=self.n_coeffs, increasing=True)
        
    def set_b(self, i):
        self.b[6 * i + 3] = self.initial_waypoints[i]

    def set_derivatives(self, i):
        self.A[6 * i + 3, 6 * i + 0] = self.T[i][0]
        self.A[6 * i + 3, 6 * i + 1] = self.T[i][1]
        self.A[6 * i + 3, 6 * i + 2] = self.T[i][2]
        self.A[6 * i + 3, 6 * i + 3] = self.T[i][3]
        self.A[6 * i + 3, 6 * i + 4] = self.T[i][4]
        self.A[6 * i + 3, 6 * i + 5] = self.T[i][5]
        self.A[6 * i + 4, 6 * i + 0] = self.T[i][0]
        self.A[6 * i + 4, 6 * i + 1] = self.T[i][1]
        self.A[6 * i + 4, 6 * i + 2] = self.T[i][2]
        self.A[6 * i + 4, 6 * i + 3] = self.T[i][3]
        self.A[6 * i + 4, 6 * i + 4] = self.T[i][4]
        self.A[6 * i + 4, 6 * i + 5] = self.T[i][5]
        self.A[6 * i + 4, 6 * i + 6] = -1.0
        self.A[6 * i + 5, 6 * i + 1] = 1 * self.T[i][0]
        self.A[6 * i + 5, 6 * i + 2] = 2 * self.T[i][1]
        self.A[6 * i + 5, 6 * i + 3] = 3 * self.T[i][2]
        self.A[6 * i + 5, 6 * i + 4] = 4 * self.T[i][3]
        self.A[6 * i + 5, 6 * i + 5] = 5 * self.T[i][4]
        self.A[6 * i + 5, 6 * i + 7] = -1.0
        self.A[6 * i + 6, 6 * i + 2] = 2 * self.T[i][0]
        self.A[6 * i + 6, 6 * i + 3] = 6 * self.T[i][1]
        self.A[6 * i + 6, 6 * i + 4] = 12 * self.T[i][2]
        self.A[6 * i + 6, 6 * i + 5] = 20 * self.T[i][3]
        self.A[6 * i + 6, 6 * i + 8] = -2.0
        self.A[6 * i + 7, 6 * i + 3] = 6.0 * self.T[i][0]
        self.A[6 * i + 7, 6 * i + 4] = 24.0 * self.T[i][1]
        self.A[6 * i + 7, 6 * i + 5] = 60.0 * self.T[i][2]
        self.A[6 * i + 7, 6 * i + 9] = -6.0
        self.A[6 * i + 8, 6 * i + 4] = 24.0 * self.T[i][0]
        self.A[6 * i + 8, 6 * i + 5] = 120.0 * self.T[i][1]
        self.A[6 * i + 8, 6 * i + 10] = -24.0

    def set_final_A(self):
        self.A[-3, -6] = 1.0
        self.A[-3, -5] = self.T[-1][1]
        self.A[-3, -4] = self.T[-1][2]
        self.A[-3, -3] = self.T[-1][3]
        self.A[-3, -2] = self.T[-1][4]
        self.A[-3, -1] = self.T[-1][5]
        self.A[-2, -5] = 1.0
        self.A[-2, -4] = 2 * self.T[-1][1]
        self.A[-2, -3] = 3 * self.T[-1][2]
        self.A[-2, -2] = 4 * self.T[-1][3]
        self.A[-2, -1] = 5 * self.T[-1][4]
        self.A[-1, -4] = 2.0
        self.A[-1, -3] = 6 * self.T[-1][1]
        self.A[-1, -2] = 12 * self.T[-1][2]
        self.A[-1, -1] = 20 * self.T[-1][3]

    def get_pos(self, t):
        '''
        get position at time t
        return a (1,D) array
        '''
        if t > sum(self.durations):
            return self.get_pos(sum(self.durations))

        if self.coefficients is None:
            self.get_coefficients(self.initial_waypoints, self.durations)

        # Locate piece index
        piece_idx = 0
        while sum(self.durations[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.durations[:piece_idx])

        c_block = self.coefficients[2*self.degrees*piece_idx:2*self.degrees*(piece_idx+1), :]

        beta = np.array([1, T, T**2, T**3, T**4, T**5])

        return np.dot(c_block.T, beta.T)

    def get_vel(self, t):
        '''
        get velocity at time t
        return a (1,D) array
        '''
        if t > sum(self.durations):
            return self.get_vel(sum(self.durations))

        if self.coefficients == []:
            self.get_coefficients(self.initial_waypoints, self.durations)

        # Locate piece index
        piece_idx = 0
        while sum(self.durations[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.durations[:piece_idx])

        c_block = self.coefficients[2*self.degrees*piece_idx:2*self.degrees*(piece_idx+1), :]

        beta = np.array([0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_acc(self, t):
        '''
        get acceleration at time t
        return a (1,D) array
        '''
        if t > sum(self.durations):
            return self.get_acc(sum(self.durations))

        if self.coefficients == []:
            self.get_coefficients(self.initial_waypoints, self.durations)

        # Locate piece index
        piece_idx = 0
        while sum(self.durations[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.durations[:piece_idx])

        c_block = self.coefficients[2*self.degrees*piece_idx:2*self.degrees*(piece_idx+1), :]

        beta = np.array([0, 0, 2, 6*T, 12*T**2, 20*T**3])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_jerk(self, t):
        '''
        get jerk at time t
        return a (1,D) array
        '''
        if t > sum(self.durations):
            return self.get_jerk(sum(self.durations))

        if self.coefficients == []:
            self.get_coefficients(self.initial_waypoints, self.durations)

        # Locate piece index
        piece_idx = 0
        while sum(self.durations[:piece_idx+1]) < t:
            piece_idx += 1

        T = t - sum(self.durations[:piece_idx])

        c_block = self.coefficients[2*self.degrees*piece_idx:2*self.degrees*(piece_idx+1), :]

        beta = np.array([0, 0, 0, 6, 24*T, 60*T**2])

        return np.dot(c_block.T, np.array([beta]).T).T

    def get_pos_array(self):
        '''
        return the full pos array
        '''
        if self.coefficients is None:
            self.get_coefficients(self.initial_waypoints, self.durations)

        t_samples = np.arange(0, sum(self.durations), 0.1)
        pos_array = np.zeros((t_samples.shape[0], self.dimensions))
        for i in range(t_samples.shape[0]):
            pos_array[i] = self.get_pos(t_samples[i])

        return pos_array

    def get_vel_array(self):
        '''
        return the full vel array
        '''
        if self.coefficients == []:
            self.get_coefficients(self.initial_waypoints, self.durations)

        t_samples = np.arange(0, sum(self.durations), 0.1)
        vel_array = np.zeros((t_samples.shape[0], self.dimensions))
        for i in range(t_samples.shape[0]):
            vel_array[i] = self.get_vel(t_samples[i])

        return vel_array

    def get_acc_array(self):
        '''
        return the full acc array
        '''
        if self.coefficients == []:
            self.get_coefficients(self.initial_waypoints, self.durations)

        t_samples = np.arange(0, sum(self.durations), 0.1)
        acc_array = np.zeros((t_samples.shape[0], self.dimensions))
        for i in range(t_samples.shape[0]):
            acc_array[i] = self.get_acc(t_samples[i])

        return acc_array

    def get_jer_array(self):
        '''
        return the full jer array
        '''
        if self.coefficients == []:
            self.get_coefficients(self.initial_waypoints, self.durations)

        t_samples = np.arange(0, sum(self.durations), 0.1)
        jer_array = np.zeros((t_samples.shape[0], self.dimensions))
        for i in range(t_samples.shape[0]):
            jer_array[i] = self.get_jerk(t_samples[i])

        return jer_array

if __name__ == "__main__":
    trajen = MinJerkPlanner()
    plotter = Plotter()
    waypoints = np.array([[1.93, -1.45, 1.0], [1.55, -1.63, 1.0], [0.8, -2.0, 1.0], [0.2, -1.55, 1.0],[-0.08, -1.38, 1.0], [-0.6, -1.03, 1.0], [-1.25, -0.58, 1.0], [-2.15, 0.03, 1.0], [-1.63, 0.53, 1.0], [-1.18, 1.03, 1.0], [-0.35, 1.4, 1.0], [0.13, 1.63, 1.0], [0.75, 1.95, 1.0], [1.3, 1.53, 1.0], [1.73, 1.03, 1.0], [1.25, 0.43, 1.0], [0.8, 0.03, 1.0], [0.83, -0.4, 1.0], [0.83, -0.95, 1.0], [1.7, -0.98, 1.0]])
    head_state = np.vstack(([2.48, -1.08, 1.0], np.zeros(3), np.zeros(3)))
    tail_state = np.vstack(([2.48, -1.08, 1.0], np.zeros(3), np.zeros(3)))
    durations = np.array([1, 1, 1, 2, 2, 2, 2, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.7, 0.6, 1, 0.3, 0.3, 0.2, 1, 0.5])
    trajen.plan(3, head_state, tail_state, waypoints, durations)
    trajectory = trajen.get_pos_array()
    plotter.initialize(trajectory, waypoints, durations, Derivative.JERK)
    plotter.plot_3D(overlap_plot=True, show_plot=True, save_plot=False)
    coefficients = trajen.coefficients
    plotter.save_data("minimal_jerk_trajectory_for_pipeline.csv", trajectory)
    plotter.save_data("minimal_jerk_coefficients_for_pipeline.csv", coefficients)