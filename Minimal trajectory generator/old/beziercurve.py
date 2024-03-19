import numpy as np
import sympy as sym
from beziercontrolpointgenerator import BezierControlPointGenerator
from polynomialgenerator import PolynomialGenerator

class BezierCurve():
    def __init__(self, waypoints, degrees = 3, resolutions=100):
        self.polytrajgen = PolynomialGenerator()
        self.waypoints = np.array(waypoints)
        self.dimensions = self.waypoints.shape[1]
        self.number_of_waypoints = self.waypoints.shape[0] - 1
        self.degrees = degrees
        self.resulutions = resolutions
        self.t_values = np.linspace(0, 1, (resolutions + 1))
        self.__control_points = np.array([]).reshape(0, self.dimensions)
        self.generate_control_points()
        self.create_T_vector()
        self.calculate_basis_fuctions()

    def generate_control_points(self):
        self.bezierpointgen = BezierControlPointGenerator()
        self.bezierpointgen.set_degrees(self.degrees)
        for i in range(self.number_of_waypoints):  # Adjusting the range to avoid index out of range
            start, end = self.waypoints[i], self.waypoints[i + 1]
            self.bezierpointgen.generate_control_points(start, end)
            control_points = self.bezierpointgen.get_control_points()
            if i < self.number_of_waypoints - 1:  # Adjusting condition to correctly check for the last waypoint
                control_points = control_points[:-1]  # Remove the last element if not the last waypoint
            self.__control_points = np.concatenate((self.__control_points, control_points))

    def create_T_vector(self):
        self.__T = np.array([self.t_values**i for i in range(self.degrees + 1)]).T

    def get_T_vector(self, derivative = 0):
        return self.__T[:, derivative]
        
    def calculate_basis_fuctions(self, C_cont=1):
        if C_cont > self.degrees:
            raise ValueError("C Continouity cannot exceed degrees.")
        t = sym.symbols('t')
        local_control_points = []
        basis_function = [t**i for i in range(self.degrees + 1)]
        matrix = sym.symbols('matrix')
        [local_control_points.append(sym.symbols(f'P{i}')) for i in range(self.degrees + 1)]
        print(self.__control_points)
        for i in range(self.degrees):
            control_points = self.get_control_points_from_segment(i)
            pass

    def get_control_points_from_segment(self, segment) -> np.ndarray:
        if segment < self.number_of_waypoints:
            return self.__control_points[segment * self.degrees : (segment + 1) * self.degrees + 1]
        raise ValueError("Segment exceeds the number of waypoints.")

if __name__ == '__main__':
    waypoints = np.array([[0, 0], [10, 10], [20, 15], [30, 25]])
    bezier_curve = BezierCurve(waypoints)
