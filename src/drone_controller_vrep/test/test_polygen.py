import unittest
from polynomialgenerator import PolynomialGenerator
from header_file import *

class TestPolynomialGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = PolynomialGenerator()

    def test_initialization(self):
        result = self.generator.generate_polynomial()
        expected_result = ensure_numpy_array(1)
        assert_array_equal(result, expected_result)

    def test_T_0(self):

        self.generator.set_degrees(Degree.CONSTANT)
        result = self.generator.generate_polynomial()
        expected_result = self._extracted_from_test_derivative_0_5(1, result)
        result = self.generator.generate_polynomial()
        expected_result = ensure_numpy_array([0, 1])
        wrong_result = ensure_numpy_array([1, 0])
        self.assertTrue(np.array_equal(result, expected_result))
        self.assertFalse(np.array_equal(result, wrong_result))

        self.generator.set_degrees(Degree.QUADRATIC)
        result = self.generator.generate_polynomial()
        expected_result = np.array([0, 0, 1])
        self.assertTrue(np.array_equal(result, expected_result))

        self.generator.set_degrees(Degree.QUINTIC)
        result = self.generator.generate_polynomial()
        expected_result = np.array([0, 0, 0, 0, 0, 1])
        self.assertTrue(np.array_equal(result, expected_result))
        
    def test_T_1(self):
        self._extracted_from_test_T_2_2(1, 1)
        
    def test_T_2(self):
        self._extracted_from_test_T_2_2(2, 4)

    def _extracted_from_test_T_2_2(self, T, arg1):
        self.generator.set_degrees(Degree.CONSTANT)
        result = self.generator.generate_polynomial(T=T)
        expected_result = self._extracted_from_test_derivative_0_5(1, result)
        result = self.generator.generate_polynomial(T=T)
        expected_result = self._extracted_from_test_derivative_0_8(T, 1, result)
        result = self.generator.generate_polynomial(T=T)
        expected_result = ensure_numpy_array([arg1, T, 1])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_derivative_0(self):
        self.generator.set_degrees(Degree.CONSTANT)
        self.generator.set_derivative(Derivative.VELOCITY)
        result = self.generator.generate_polynomial()
        expected_result = self._extracted_from_test_derivative_0_5(0, result)
        self.generator.set_derivative(Derivative.VELOCITY)
        result = self.generator.generate_polynomial()
        expected_result = self._extracted_from_test_derivative_0_8(1, 0, result)
        result = self.generator.generate_polynomial(derivative=Derivative.VELOCITY)
        expected_result = ensure_numpy_array([0, 1, 0])
        self.assertTrue(np.array_equal(result, expected_result))
        result = self.generator.generate_polynomial(derivative=Derivative.ACCELERATION)
        expected_result = ensure_numpy_array([2, 0, 0])
        self.assertTrue(np.array_equal(result, expected_result))

    def _extracted_from_test_derivative_0_8(self, arg0, arg1, result):
        result = ensure_numpy_array([arg0, arg1])
        self.assertTrue(np.array_equal(result, result))
        self.generator.set_degrees(Degree.QUADRATIC)
        return result

    def _extracted_from_test_derivative_0_5(self, arg0, result):
        result = ensure_numpy_array(arg0)
        self.assertTrue(np.array_equal(result, result))
        self.generator.set_degrees(Degree.LINEAR)
        return result
    
    def test_derivative_1(self):
        result = self.generator.generate_polynomial(Degree.SEPTIC, Derivative.VELOCITY, 1)
        expected_result = ensure_numpy_array([7, 6, 5, 4, 3, 2, 1, 0])
        self.assertTrue(np.array_equal(result, expected_result))
        
        result = self.generator.generate_polynomial(Degree.SEPTIC, Derivative.ACCELERATION, 0)
        expected_result = ensure_numpy_array([0,0,0,0,0,2,0,0])
        self.assertTrue(np.array_equal(result, expected_result))
        
        result = self.generator.generate_polynomial(Degree.SEPTIC, Derivative.ACCELERATION, 1)
        expected_result = ensure_numpy_array([42,30,20,12,6,2,0,0])
        self.assertTrue(np.array_equal(result, expected_result))
        
        result = self.generator.generate_polynomial(Degree.SEPTIC, Derivative.ACCELERATION, 2)
        expected_result = ensure_numpy_array([42*2**5,30*2**4,20*2**3,12*2**2,6*2**1,2*2**0,0,0])
        self.assertTrue(np.array_equal(result, expected_result))

if __name__ == '__main__':
    unittest.main()