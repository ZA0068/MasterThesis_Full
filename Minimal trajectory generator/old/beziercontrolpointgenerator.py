from ..src.header_file import *

class BezierControlPointGenerator:
    def __init__(self, start=None, stop=None, degrees=1):
        if start is None or stop is None or degrees < 1:
            return
        self.set_degrees(degrees)
        self.generate_control_points(start, stop)

    def generate_control_points(self, start, stop):
        self.__start = ensure_numpy_array(start)
        self.__stop = ensure_numpy_array(stop)
        self.__generate()

    def set_degrees(self, degrees):
        if degrees < 1:
            raise ValueError("Degree must be at least 1.")
        self.__degrees = degrees

    def __generate(self):
        if self.__degrees == 1:
            self.__control_points = np.array([self.__start, self.__stop])
        else:
            spacing = 1 / self.__degrees
            self.__control_points = np.array([
                self.__start if i == 0 else 
                self.__stop if i == self.__degrees else 
                (1 - i * spacing) * self.__start + i * spacing * self.__stop
                for i in range(self.__degrees + 1)
            ])

    def get_control_points(self) -> np.ndarray:
        return self.__control_points

    def set_control_points(self, control_points: np.ndarray):
        self.__control_points = np.array(control_points)
        
    def set_control_point_index(self, index: int, control_point: np.ndarray):
        if index > self.__degrees:
            raise ValueError("Index exceeds the number of control points.")
        elif index < 0:
            raise ValueError("Index must be at least 0.")
        self.__control_points[index] = np.array(control_point)

# Example usage
def main():
    start = [0, 0]
    stop = [10, 10]
    degree = 3
    bezier = BezierControlPointGenerator(start, stop, degree)
    print("Control Points:", bezier.get_control_points())
    
if __name__ == "__main__":
    main()
