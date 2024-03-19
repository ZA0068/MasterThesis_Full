import pytest
from polynomialtrajectory import MinimalTrajectoryGenerator
from header_file import *


@pytest.fixture
def plotter():
    return Plotter()

class TestMinimalAccelerationTrajectorySimple:
    @pytest.fixture
    def order(self):
        return Derivative.ACCELERATION
    
    @pytest.fixture
    def waypoints(self):
        return np.array([[0, 0], [10, 10], [20, 15]])
    
    @pytest.fixture
    def durations(self):
        return np.array([1, 1])

    @pytest.fixture
    def mintrajgen(self):
        return MinimalTrajectoryGenerator()

       
    def test_spline_parameters(self, mintrajgen, waypoints, durations):
        mintrajgen.initialize_spline_parameters(waypoints, durations)
        assert_array_equal(mintrajgen.get_waypoints(), waypoints)
        assert_array_equal(mintrajgen.get_durations(), durations)
        assert_equal(mintrajgen.get_number_of_waypoints(), 1)
        assert_equal(mintrajgen.get_number_of_splines(), 2)
        assert_equal(mintrajgen.get_dimensions(), 2)
        assert_array_equal(mintrajgen.get_splines(), np.array([]))

    def test_polynomial_parameters(self, mintrajgen, order):
        mintrajgen.initialize_polynomial_parameters(order)
        assert_equal(mintrajgen.get_derivative(), order)
        assert_equal(mintrajgen.get_degrees(), 3)
        assert_equal(mintrajgen.get_number_of_coefficients(), 4)

    def test_set_start_point(self, mintrajgen, waypoints, durations):
        mintrajgen.initialize_spline_parameters(waypoints, durations)
        assert_equal(mintrajgen.get_start_point(), np.array([0, 0]))
        mintrajgen.set_start_point([50, 50])
        assert_equal(mintrajgen.get_start_point(), np.array([50, 50]))
    
    def test_set_end_point(self, mintrajgen, waypoints, durations):
        mintrajgen.initialize_spline_parameters(waypoints, durations)
        assert_equal(mintrajgen.get_end_point(), np.array([20, 15]))
        mintrajgen.set_end_point([50, 50])
        assert_equal(mintrajgen.get_end_point(), np.array([50, 50]))
    
    @pytest.fixture
    def mintrajgen_param(self, mintrajgen, waypoints, durations, order):
        mintrajgen.initialize_spline_parameters(waypoints, durations)
        mintrajgen.initialize_polynomial_parameters(order)
        return mintrajgen

    @pytest.fixture
    def expected_duration_from_velocity(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 1.0

    def test_duration_from_velocity(self, mintrajgen_param, expected_duration_from_velocity):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen_param, 1.0, expected_duration_from_velocity
        )

    @pytest.fixture
    def expected_duration_from_velocity_2(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 3.5

    def test_duration_from_velocity_2(self, mintrajgen_param, expected_duration_from_velocity_2):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen_param, 3.5, expected_duration_from_velocity_2
        )

    def __extracted_from_test_duration_from_velocity_2_2(self, mintrajgen_param, arg1, arg2):
        mintrajgen_param.set_maximum_velocity(arg1)
        mintrajgen_param.set_time_based_on_velocity()
        assert_array_equal(mintrajgen_param.get_durations(), arg2)

    def test_AB_matrices(self, mintrajgen_param):
        mintrajgen_param.init_AB_matrices()
        assert_array_equal(mintrajgen_param.A_matrix().shape, (8, 8))
        assert_array_equal(mintrajgen_param.b_matrix().shape, (8, 2))

    @pytest.fixture
    def expected_A(self):
        return np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [3, 2, 1, 0, 0, 0, -1, 0],
            [6, 2, 0, 0, 0, -2, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 6, 2, 0, 0]
        ])
        
    def test_generate_position_constraints_4_A(self, mintrajgen_param, expected_A):
        mintrajgen_param.init_AB_matrices()
        mintrajgen_param.generate_position_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen_param, 0, 4, expected_A
        )
        

    def test_generate_continuity_4_A(self, mintrajgen_param, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen_param
        )
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen_param, 4, 6, expected_A
        )
        

    def test_generate_start_and_goal_constraints_4_A(self, mintrajgen_param, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen_param
        )
        mintrajgen_param.generate_start_and_goal_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen_param, 6, 8, expected_A
        )

    def _extracted_from_test_generate_start_and_goal_constraints_4_A_2(self, mintrajgen_param):
        mintrajgen_param.init_AB_matrices()
        mintrajgen_param.generate_position_constraints()
        mintrajgen_param.generate_continuity_constraints()

    def _extracted_from_test_generate_start_and_goal_constraints_4_A_4(self, mintrajgen_param, arg1, arg2, expected_A):
        result = mintrajgen_param.A_matrix()[arg1:arg2, :]
        assert_array_equal(result, expected_A[arg1:arg2, :])
        result = mintrajgen_param.get_nth_row()
        assert_equal(result, arg2)

    @pytest.fixture
    def expected_B(self):
        return np.array([
            [0, 0],
            [10, 10],
            [10, 10],
            [20, 15],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        
    def test_generate_position_constraints_4_b(self, mintrajgen_param, expected_B):
        mintrajgen_param.init_AB_matrices()
        mintrajgen_param.generate_position_constraints()
        assert_array_equal(mintrajgen_param.b_matrix(), expected_B)

    @pytest.fixture
    def mintrajgen_fin(self, mintrajgen_param):
        mintrajgen_param.set_maximum_velocity(100)
        mintrajgen_param.set_dt(0.01)
        mintrajgen_param.create_poly_matrices()
        return mintrajgen_param

    def filename(self, type):
        return f"minimal_acceleration_{type}.csv"

    def test_compute_splines(self, mintrajgen_fin, waypoints, durations, plotter):
        mintrajgen_fin.compute_splines()
        data = mintrajgen_fin.get_splines()
        plotter.initialize(data, waypoints, durations)
        plotter.plot_2D(save_plot=True)
        plotter.save_data(self.filename('trajectory'), data)
        plotter.save_data(self.filename('waypoints'), waypoints)
        plotter.save_data(self.filename('durations'), durations)

    def test_plot_splines(self, plotter):
        data = plotter.read_data(self.filename('trajectory'))
        waypoints = plotter.read_data(self.filename('waypoints'))
        durations = plotter.read_data(self.filename('durations'))
        plotter.initialize(data, waypoints, durations, Derivative.ACCELERATION)
        plotter.plot_2D()
        plotter.plot_time_data_individually(save_plot=True)

class TestMinimalTrajectoryGeneratorSimpleExtended():
    
    @pytest.fixture
    def order(self):
        return Derivative.ACCELERATION
    
    @pytest.fixture
    def waypoints(self):
        return np.array([[0, 0], [10, 10], [20, 15], [30, 25]])
    
    @pytest.fixture
    def durations(self):
        return np.array([1, 1, 1])
    
    @pytest.fixture
    def velocity(self):
        return 100
    
    @pytest.fixture
    def dt(self):
        return 0.01

    @pytest.fixture
    def mintrajgen(self, waypoints, durations, order, velocity, dt):
        mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, order)
        mintrajgen.set_maximum_velocity(velocity)
        mintrajgen.set_dt(dt)
        return mintrajgen

    def test_bezier_splines(self, mintrajgen, waypoints, durations):
        assert_array_equal(mintrajgen.get_waypoints(), waypoints)
        assert_array_equal(mintrajgen.get_durations(), durations)
        assert_array_equal(mintrajgen.get_time_matrix(), [0, 1, 2, 3])
        assert_equal(mintrajgen.get_number_of_waypoints(), 2)
        assert_equal(mintrajgen.get_number_of_splines(), 3)
        assert_equal(mintrajgen.get_dimensions(), 2)
        assert_array_equal(mintrajgen.get_splines(), np.array([]))

    def test_polynomial_parameters(self, mintrajgen, order):
        assert_equal(mintrajgen.get_derivative(), order)
        assert_equal(mintrajgen.get_degrees(), 3)
        assert_equal(mintrajgen.get_number_of_coefficients(), 4)

    def test_set_start_point(self, mintrajgen):
        assert_equal(mintrajgen.get_start_point(), np.array([0, 0]))
        mintrajgen.set_start_point([50, 50])
        assert_equal(mintrajgen.get_start_point(), np.array([50, 50]))
    
    def test_set_end_point(self, mintrajgen):
        assert_equal(mintrajgen.get_end_point(), np.array([30, 25]))
        mintrajgen.set_end_point([50, 50])
        assert_equal(mintrajgen.get_end_point(), np.array([50, 50]))

    @pytest.fixture
    def expected_duration_from_velocity(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 1.0

    def test_duration_from_velocity(self, mintrajgen, expected_duration_from_velocity):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen, 1.0, expected_duration_from_velocity
        )


    @pytest.fixture
    def expected_duration_from_velocity_2(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 3.5

    def test_duration_from_velocity_2(self, mintrajgen, expected_duration_from_velocity_2):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen, 3.5, expected_duration_from_velocity_2
        )

    def __extracted_from_test_duration_from_velocity_2_2(self, mintrajgen_param, arg1, arg2):
        mintrajgen_param.set_maximum_velocity(arg1)
        mintrajgen_param.set_time_based_on_velocity()
        assert_array_equal(mintrajgen_param.get_durations(), arg2)

    def test_AB_matrices(self, mintrajgen):
        mintrajgen.init_AB_matrices()
        assert_array_equal(mintrajgen.A_matrix().shape, (12, 12))
        assert_array_equal(mintrajgen.b_matrix().shape, (12, 2))

    @pytest.fixture
    def expected_A(self):
        return np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [3, 2, 1, 0, 0, 0,-1, 0,  0, 0, 0, 0], 
            [0, 0, 0, 0, 3,  2,  1, 0, 0,0,-1, 0],
            [6, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 6,  2, 0, 0, 0, -2, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 6, 2, 0, 0]
        ])
        
    def _extracted_from_test_generate_start_and_goal_constraints_4_A_2(self, mintrajgen_param):
        mintrajgen_param.init_AB_matrices()
        mintrajgen_param.generate_position_constraints()
        mintrajgen_param.generate_continuity_constraints()

    def _extracted_from_test_generate_start_and_goal_constraints_4_A_4(self, mintrajgen, arg1, arg2, expected_A):
        result = mintrajgen.A_matrix()[arg1:arg2, :]
        assert_array_equal(result, expected_A[arg1:arg2, :])
        result = mintrajgen.get_nth_row()
        assert_equal(result, arg2)
        
    def test_generate_position_constraints_4_A(self, mintrajgen, expected_A):
        mintrajgen.init_AB_matrices()
        mintrajgen.generate_position_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 0, 6, expected_A
        )

    def test_generate_continuity_4_A(self, mintrajgen, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen
        )
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 6, 10, expected_A
        )

    def test_generate_start_and_goal_constraints_4_A(self, mintrajgen, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen
        )
        mintrajgen.generate_start_and_goal_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 10, 12, expected_A
        )

    @pytest.fixture
    def expected_B(self):
        return np.array([
            [0, 0],
            [10, 10],
            [10, 10],
            [20, 15],
            [20, 15],
            [30, 25],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ])
        
    def test_generate_position_constraints_4_b(self, mintrajgen, expected_B):
        mintrajgen.init_AB_matrices()
        mintrajgen.generate_position_constraints()
        assert_array_equal(mintrajgen.b_matrix(), expected_B)

    @pytest.fixture
    def mintrajgen_fin(self, mintrajgen):
        mintrajgen.create_poly_matrices()
        return mintrajgen

    def filename(self, type):
        return f"minimal_acceleration_{type}_extended.csv"

    
    
    def test_compute_splines(self, mintrajgen_fin, waypoints, durations, plotter):
        mintrajgen_fin.compute_splines()
        data = mintrajgen_fin.get_splines()
        plotter.initialize(data, waypoints, durations)
        plotter.append_title_name(append = "extended")
        plotter.plot_2D(save_plot=True)
        plotter.plot_time_data_individually(save_plot=True)
        plotter.save_data(self.filename('trajectory'), data)
        plotter.save_data(self.filename('waypoints'), waypoints)
        plotter.save_data(self.filename('durations'), durations)

    def test_plot_splines(self, plotter):
        data = plotter.read_data(self.filename('trajectory'))
        waypoints = plotter.read_data(self.filename('waypoints'))
        durations = plotter.read_data(self.filename('durations'))
        plotter.initialize(data, waypoints, durations, Derivative.ACCELERATION)
        plotter.append_title_name(append = "extended")
        plotter.plot_time_data_at_same_time(save_plot=True)

class TestMinimalTrajectoryGeneratorSimpleExtended3D():
    @pytest.fixture
    def order(self):
        return Derivative.ACCELERATION
    
    @pytest.fixture
    def waypoints(self):
        return np.array([[0, 0, 10], [10, 10, 10], [20, 15, 25], [30, 25, 10]])
    
    @pytest.fixture
    def durations(self):
        return np.array([2, 4, 6])
    
    @pytest.fixture
    def velocity(self):
        return 100
    
    @pytest.fixture
    def dt(self):
        return 0.01

    @pytest.fixture
    def mintrajgen(self, waypoints, durations, order, velocity, dt):
        mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, order)
        mintrajgen.set_maximum_velocity(velocity)
        mintrajgen.set_dt(dt)
        return mintrajgen

    def test_bezier_splines(self, mintrajgen, waypoints, durations):
        assert_array_equal(mintrajgen.get_waypoints(), waypoints)
        assert_array_equal(mintrajgen.get_durations(), durations)
        assert_array_equal(mintrajgen.get_time_matrix(), [0, 2, 6, 12])
        assert_equal(mintrajgen.get_number_of_waypoints(), 2)
        assert_equal(mintrajgen.get_number_of_splines(), 3)
        assert_equal(mintrajgen.get_dimensions(), 3)
        assert_array_equal(mintrajgen.get_splines(), np.array([]))

    def test_polynomial_parameters(self, mintrajgen, order):
        assert_equal(mintrajgen.get_derivative(), order)
        assert_equal(mintrajgen.get_degrees(), 3)
        assert_equal(mintrajgen.get_number_of_coefficients(), 4)

    def test_set_start_point(self, mintrajgen):
        assert_equal(mintrajgen.get_start_point(), np.array([0, 0, 10]))
        mintrajgen.set_start_point([50, 50, 50])
        assert_equal(mintrajgen.get_start_point(), np.array([50, 50, 50]))
    
    def test_set_end_point(self, mintrajgen):
        assert_equal(mintrajgen.get_end_point(), np.array([30, 25, 10]))
        mintrajgen.set_end_point([50, 50, 50])
        assert_equal(mintrajgen.get_end_point(), np.array([50, 50, 50]))


    @pytest.fixture
    def expected_duration_from_velocity(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 1.0

    def test_duration_from_velocity(self, mintrajgen, expected_duration_from_velocity):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen, 1.0, expected_duration_from_velocity
        )


    @pytest.fixture
    def expected_duration_from_velocity_2(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 3.5

    def test_duration_from_velocity_2(self, mintrajgen, expected_duration_from_velocity_2):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen, 3.5, expected_duration_from_velocity_2
        )

    def __extracted_from_test_duration_from_velocity_2_2(self, mintrajgen_param, arg1, arg2):
        mintrajgen_param.set_maximum_velocity(arg1)
        mintrajgen_param.set_time_based_on_velocity()
        assert_array_almost_equal(mintrajgen_param.get_durations(), arg2, 0.01)

    def test_AB_matrices(self, mintrajgen):
        mintrajgen.init_AB_matrices()
        assert_array_equal(mintrajgen.A_matrix().shape, (12, 12))
        assert_array_equal(mintrajgen.b_matrix().shape, (12, 3))

    @pytest.fixture
    def expected_A(self):
        return np.array([
            [0, 0, 0, 1,  0,  0, 0, 0, 0, 0, 0, 0],
            [8, 4, 2, 1,  0,  0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0,  0,  0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 64, 16, 4, 1, 0, 0, 0, 0],
            [0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0,  0,  0, 0, 0,216,36,6, 1],
            [12, 4, 1, 0, 0,  0,-1, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 48,  8, 1, 0, 0, 0,-1, 0],
            [12, 2, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 24, 2, 0, 0, -0, -2, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 36, 2, 0, 0]
        ])
        
    def _extracted_from_test_generate_start_and_goal_constraints_4_A_2(self, mintrajgen_param):
        mintrajgen_param.init_AB_matrices()
        mintrajgen_param.generate_position_constraints()
        mintrajgen_param.generate_continuity_constraints()

    def _extracted_from_test_generate_start_and_goal_constraints_4_A_4(self, mintrajgen, arg1, arg2, expected_A):
        result = mintrajgen.A_matrix()[arg1:arg2, :]
        assert_array_equal(result, expected_A[arg1:arg2, :])
        result = mintrajgen.get_nth_row()
        assert_equal(result, arg2)
        
    def test_generate_position_constraints_4_A(self, mintrajgen, expected_A):
        mintrajgen.init_AB_matrices()
        mintrajgen.generate_position_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 0, 6, expected_A
        )

    def test_generate_continuity_4_A(self, mintrajgen, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen
        )
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 6, 10, expected_A
        )

    def test_generate_start_and_goal_constraints_4_A(self, mintrajgen, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen
        )
        mintrajgen.generate_start_and_goal_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 10, 12, expected_A
        )

    @pytest.fixture
    def expected_B(self):
        return np.array([
            [0, 0, 10],
            [10, 10, 10],
            [10, 10, 10],
            [20, 15, 25],
            [20, 15, 25],
            [30, 25, 10],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        
    def test_generate_position_constraints_4_b(self, mintrajgen, expected_B):
        mintrajgen.init_AB_matrices()
        mintrajgen.generate_position_constraints()
        assert_array_equal(mintrajgen.b_matrix(), expected_B)

    @pytest.fixture
    def mintrajgen_fin(self, mintrajgen):
        mintrajgen.create_poly_matrices()
        return mintrajgen

    def filename(self, type):
        return f"minimal_acceleration_{type}_extended3D.csv"

    def test_compute_splines(self, mintrajgen_fin, waypoints, durations, plotter):
        mintrajgen_fin.compute_splines()
        data = mintrajgen_fin.get_splines()
        plotter.initialize(data, waypoints, durations)
        plotter.plot_3D()
        plotter.save_data(self.filename('trajectory'), data)
        plotter.save_data(self.filename('waypoints'), waypoints)
        plotter.save_data(self.filename('durations'), durations)

    def test_plot_splines(self, plotter):
        data = plotter.read_data(self.filename('trajectory'))
        waypoints = plotter.read_data(self.filename('waypoints'))
        durations = plotter.read_data(self.filename('durations'))
        plotter.initialize(data, waypoints, durations, Derivative.ACCELERATION)
        plotter.plot_3D()
        plotter.plot_time_data_at_same_time(save_plot=True)
        plotter.plot_time_data_individually(save_plot=True)

class TestMinimalTrajectoryGeneratorSimpleJerk():
    @pytest.fixture
    def order(self):
        return Derivative.JERK
    
    @pytest.fixture
    def waypoints(self):
        return np.array([[0, 0, 10], [10, 10, 10], [20, 15, 25], [30, 25, 10]])
    
    @pytest.fixture
    def durations(self):
        return np.array([2, 4, 6])
    
    @pytest.fixture
    def velocity(self):
        return 100
    
    @pytest.fixture
    def dt(self):
        return 0.01

    @pytest.fixture
    def mintrajgen(self, waypoints, durations, order, velocity, dt):
        mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, order)
        mintrajgen.set_maximum_velocity(velocity)
        mintrajgen.set_dt(dt)
        return mintrajgen

    def test_bezier_splines(self, mintrajgen, waypoints, durations):
        assert_array_equal(mintrajgen.get_waypoints(), waypoints)
        assert_array_equal(mintrajgen.get_durations(), durations)
        assert_array_equal(mintrajgen.get_time_matrix(), [0, 2, 6, 12])
        assert_equal(mintrajgen.get_number_of_waypoints(), 2)
        assert_equal(mintrajgen.get_number_of_splines(), 3)
        assert_equal(mintrajgen.get_dimensions(), 3)
        assert_array_equal(mintrajgen.get_splines(), np.array([]))

    def test_polynomial_parameters(self, mintrajgen, order):
        assert_equal(mintrajgen.get_derivative(), order)
        assert_equal(mintrajgen.get_degrees(), 5)
        assert_equal(mintrajgen.get_number_of_coefficients(), 6)

    def test_set_start_point(self, mintrajgen):
        assert_equal(mintrajgen.get_start_point(), np.array([0, 0, 10]))
        mintrajgen.set_start_point([50, 50, 50])
        assert_equal(mintrajgen.get_start_point(), np.array([50, 50, 50]))
    
    def test_set_end_point(self, mintrajgen):
        assert_equal(mintrajgen.get_end_point(), np.array([30, 25, 10]))
        mintrajgen.set_end_point([50, 50, 50])
        assert_equal(mintrajgen.get_end_point(), np.array([50, 50, 50]))

    @pytest.fixture
    def expected_duration_from_velocity(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 1.0

    def test_duration_from_velocity(self, mintrajgen, expected_duration_from_velocity):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen, 1.0, expected_duration_from_velocity
        )


    @pytest.fixture
    def expected_duration_from_velocity_2(self, waypoints):
        return np.linalg.norm(np.diff(waypoints, axis=0), axis=1) / 3.5

    def test_duration_from_velocity_2(self, mintrajgen, expected_duration_from_velocity_2):
        self.__extracted_from_test_duration_from_velocity_2_2(
            mintrajgen, 3.5, expected_duration_from_velocity_2
        )

    def __extracted_from_test_duration_from_velocity_2_2(self, mintrajgen_param, arg1, arg2):
        mintrajgen_param.set_maximum_velocity(arg1)
        mintrajgen_param.set_time_based_on_velocity()
        assert_array_almost_equal(mintrajgen_param.get_durations(), arg2, 0.01)

    def test_AB_matrices(self, mintrajgen):
        mintrajgen.init_AB_matrices()
        assert_array_equal(mintrajgen.A_matrix().shape, (18, 18))
        assert_array_equal(mintrajgen.b_matrix().shape, (18, 3))

    @pytest.fixture
    def expected_A(self):
        return np.array([
            [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [32,16,8,4,2,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,1024,256,64,16,4,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,7776,1296,216,36,6,1],
            [80,32,12,4,1,0,0,0,0,0,-1,0,0,0,0,0,0,0], 
            [0,0,0,0,0,0,1280,256,48,8,1,0,0,0,0,0,-1,0],
            [160,48,12,2,0,0,0,0,0,-2,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1280,192,24,2,0,0,0,0,0,-2,0,0],
            [240,48,6,0,0,0,0,0,-6,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,960,96,6,0,0,0,0,0,-6,0,0,0],
            [240,24,0,0,0,0,0,-24,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,480,24,0,0,0,0,0,-24,0,0,0,0],
            [0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,2160,144,6,0,0,0],
            [0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,720,24,0,0,0,0],
        ])
        
    def _extracted_from_test_generate_start_and_goal_constraints_4_A_2(self, mintrajgen_param):
        mintrajgen_param.init_AB_matrices()
        mintrajgen_param.generate_position_constraints()
        mintrajgen_param.generate_continuity_constraints()

    def _extracted_from_test_generate_start_and_goal_constraints_4_A_4(self, mintrajgen, arg1, arg2, expected_A):
        result = mintrajgen.A_matrix()[arg1:arg2, :]
        assert_array_equal(result, expected_A[arg1:arg2, :])
        result = mintrajgen.get_nth_row()
        assert_equal(result, arg2)
        
    def test_generate_position_constraints_4_A(self, mintrajgen, expected_A):
        mintrajgen.init_AB_matrices()
        mintrajgen.generate_position_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 0, 6, expected_A
        )

    def test_generate_continuity_4_A(self, mintrajgen, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen
        )
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 6, 14, expected_A
        )

    def test_generate_start_and_goal_constraints_4_A(self, mintrajgen, expected_A):
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_2(
            mintrajgen
        )
        mintrajgen.generate_start_and_goal_constraints()
        self._extracted_from_test_generate_start_and_goal_constraints_4_A_4(
            mintrajgen, 14, 18, expected_A
        )

    @pytest.fixture
    def expected_B(self):
        return np.array([
            [0, 0, 10],
            [10, 10, 10],
            [10, 10, 10],
            [20, 15, 25],
            [20, 15, 25],
            [30, 25, 10],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        
    def test_generate_position_constraints_4_b(self, mintrajgen, expected_B):
        mintrajgen.init_AB_matrices()
        mintrajgen.generate_position_constraints()
        assert_array_equal(mintrajgen.b_matrix(), expected_B)

    @pytest.fixture
    def mintrajgen_fin(self, mintrajgen):
        mintrajgen.create_poly_matrices()
        return mintrajgen

    def filename(self, type):
        return f"minimal_jerk_{type}_extended3D.csv"

    def test_compute_splines(self, mintrajgen_fin, waypoints, durations, plotter):
        mintrajgen_fin.compute_splines()
        data = mintrajgen_fin.get_splines()
        plotter.initialize(data, waypoints, durations)
        plotter.plot_3D()
        plotter.save_data(self.filename('trajectory'), data)
        plotter.save_data(self.filename('waypoints'), waypoints)
        plotter.save_data(self.filename('durations'), durations)

    def test_plot_splines(self, plotter):
        data = plotter.read_data(self.filename('trajectory'))
        waypoints = plotter.read_data(self.filename('waypoints'))
        durations = plotter.read_data(self.filename('durations'))
        plotter.initialize(data, waypoints, durations)
        plotter.plot_time_data_individually(save_plot=True)
        plotter.plot_time_data_at_same_time(save_plot=True)


class TestFinalMinimalTrajectoryGeneration():
    @pytest.fixture
    def order_jerk(self):
        return Derivative.JERK
    
    @pytest.fixture
    def order_snap(self):
        return Derivative.SNAP
    
    @pytest.fixture
    def order_acceleration(self):
        return Derivative.ACCELERATION
    
    @pytest.fixture
    def waypoints(self):
        return np.array([[0, 0, 10], [10, 10, 10], [20, 15, 25], [30, 25, 10], [0, 0, 10], [20, -20, 10], [40, 10, 20]])
    
    @pytest.fixture
    def durations(self):
        return np.array([5, 5, 5, 5, 5, 5])
    
    @pytest.fixture
    def velocity(self):
        return 100
    
    @pytest.fixture
    def dt(self):
        return 0.01

    @pytest.fixture
    def mintrajgen_accel(self, waypoints, durations, order_acceleration, velocity, dt):
        mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, order_acceleration)
        mintrajgen.set_maximum_velocity(velocity)
        mintrajgen.set_dt(dt)
        return mintrajgen
    
    @pytest.fixture
    def mintrajgen_jerk(self, waypoints, durations, order_jerk, velocity, dt):
        mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, order_jerk)
        mintrajgen.set_maximum_velocity(velocity)
        mintrajgen.set_dt(dt)
        return mintrajgen
    
    @pytest.fixture
    def mintrajgen_snap(self, waypoints, durations, order_snap, velocity, dt):
        mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, order_snap)
        mintrajgen.set_maximum_velocity(velocity)
        mintrajgen.set_dt(dt)
        return mintrajgen
    
    def filename(self, type):
        return f"final_minimal_{type}_trajectory_.csv"
    
    def test_save_3D_plots(self, mintrajgen_accel, mintrajgen_jerk, mintrajgen_snap, waypoints, durations, plotter):
        mintrajgen_accel.create_poly_matrices()
        mintrajgen_jerk.create_poly_matrices()
        mintrajgen_snap.create_poly_matrices()
        mintrajgen_accel.compute_splines()
        mintrajgen_jerk.compute_splines()
        mintrajgen_snap.compute_splines()
        data_accel = mintrajgen_accel.get_splines()
        data_jerk = mintrajgen_jerk.get_splines()
        data_snap = mintrajgen_snap.get_splines()

        self._test_compute_multiple_splines(
            plotter, data_accel, waypoints, durations
        )
        self._test_compute_multiple_splines(
            plotter, data_jerk, waypoints, durations
        )
        self._test_compute_multiple_splines(
            plotter, data_snap, waypoints, durations
        )
        plotter.save_data(self.filename('acceleration'), data_accel)
        plotter.save_data(self.filename('jerk'), data_jerk)
        plotter.save_data(self.filename('snap'), data_snap)
        plotter.save_data("waypoints.csv", waypoints)
        plotter.save_data("durations.csv", durations)

    def _test_compute_multiple_splines(self, plotter, arg1, waypoints, durations):
        plotter.initialize(arg1, waypoints, durations)
        plotter.plot_3D(save_plot=True)
        plotter.reset()
        
        
    def test_plot_and_save_all_time_data(self, plotter):
        data_accel = plotter.read_data(self.filename('acceleration'))
        data_jerk = plotter.read_data(self.filename('jerk'))
        data_snap = plotter.read_data(self.filename('snap'))
        waypoints = plotter.read_data("waypoints.csv")
        durations = plotter.read_data("durations.csv")

        self._plot_and_save_all_data_test_extracted(
            plotter, data_accel, waypoints, durations
        )
        self._plot_and_save_all_data_test_extracted(
            plotter, data_jerk, waypoints, durations
        )
        self._plot_and_save_all_data_test_extracted(
            plotter, data_snap, waypoints, durations
        )

    def _plot_and_save_all_data_test_extracted(self, plotter, arg1, waypoints, durations):
        plotter.initialize(arg1, waypoints, durations)
        plotter.plot_time_data_at_same_time(save_plot=True)
        plotter.plot_time_data_individually(save_plot=True)
        plotter.reset()


class TestMinimalJerkTrajectoryOptional():
    
    @pytest.fixture
    def waypoints(self):
        return np.array([[0, 0, 10], [10, 10, 10], [20, 15, 25], [30, 25, 10], [0, 0, 10], [20, -20, 10], [40, 10, 20]])
    
    @pytest.fixture
    def durations(self):
        return np.array([5, 5, 5, 5, 5, 5])
    
    @pytest.fixture
    def velocity(self):
        return 1
    
    @pytest.fixture
    def dt(self):
        return 0.01

    @pytest.fixture
    def order_jerk(self):
        return Derivative.JERK
    
    @pytest.fixture
    def mintrajgen_jerk(self, waypoints, durations, order_jerk, velocity, dt):
        mintrajgen = MinimalTrajectoryGenerator(waypoints, durations, order_jerk)
        mintrajgen.set_maximum_velocity(velocity)
        mintrajgen.set_dt(dt)
        return mintrajgen
    
    def test_jerk_optinal(self, mintrajgen_jerk, plotter):
        mintrajgen_jerk.set_max_continuity_derivative(2)
        mintrajgen_jerk.set_minmax_start_end_derivative(1, 3)
        mintrajgen_jerk.create_poly_matrices()
        mintrajgen_jerk.compute_splines()
        data = mintrajgen_jerk.get_splines()
        plotter.initialize(data, mintrajgen_jerk.get_waypoints(), mintrajgen_jerk.get_durations())
        plotter.plot_3D(save_plot=False)
        plotter.plot_time_data_at_same_time(save_plot=False)
        