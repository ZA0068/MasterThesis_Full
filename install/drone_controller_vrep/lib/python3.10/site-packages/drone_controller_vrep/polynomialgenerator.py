from header_file import *

class PolynomialGenerator:
    def __init__(self, degrees=Degree.CONSTANT, derivative=Derivative.POSITION):
        self.set_degrees(degrees)
        self.set_derivative(derivative)

    def set_degrees(self, degrees):
        degrees = self._filter_Enums(degrees)
        if degrees > -1:
            self._degrees = degrees

    def create_basis_values(self, degrees):
        deg = self._extract_and_compare_values(degrees, self._degrees)
        self.__base = np.ones(deg + 1)
        self.__exponents = np.arange(deg, -1, -1)
    
    def set_derivative(self, derivative):
        derivative = self._filter_Enums(derivative)
        if derivative > -1:
            self._derivative = derivative

    def create_polynomials(self, derivative):
        diff = self._extract_and_compare_values(derivative, self._derivative)
        self.__powers = self._shift_array(self.__exponents, -diff)
        self.__polynomial = np.pad(np.polyder(self.__base, diff), (0, diff), 'constant')

    def _extract_and_compare_values(self, derivative, default=0):
        derivative = self._filter_Enums(derivative)
        return derivative if derivative > -1 else default

    def _filter_Enums(self, derivative):
        if isinstance(derivative, Enum):
            derivative = derivative.value
        return derivative
    
    def generate_at_T(self, T=0):
        return self.__polynomial * T ** self.__powers
        
    def generate_polynomial(self, degrees=-1, derivative=-1, T=0, inverse=False):
        self.create_basis_values(degrees)
        self.create_polynomials(derivative)
        polynomial = self.generate_at_T(T)
        if inverse:
            polynomial = np.flip(polynomial)
        return polynomial

    def _shift_array(self, arr, shift):
        shifted_arr = np.roll(arr, shift)
        match shift:
            case 0:
                return shifted_arr
            case _ if shift > 0:
                shifted_arr[:shift] = 0
            case _:
                shifted_arr[shift:] = 0
        return shifted_arr

    def get_derivative(self):
        return Derivative(self._derivative)
    
    def get_degrees(self):
        return Degree(self._degrees)

if __name__ == '__main__':
    generator = PolynomialGenerator()
    print("Zeroth derivative:", generator.generate_polynomial(Degree.QUINTIC, Derivative.JERK, 1))

