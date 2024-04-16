import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3, Quaternion, Point
import rclpy.publisher
from std_msgs.msg import Float64, Bool
from tf_transformations import euler_from_quaternion
from .online_impedance_adaptive_controller import OIAC
from .propotional_integral_derivative_controller import PID
from drone_controller_vrep.header_file import read_data, Derivative, Controller, TrajectoryState
import numpy as np
from .gui import App, QApplication
from enum import Enum

class DroneFeedbackController(Node):
    @staticmethod
    def quaternion_to_euler(quaternion: Quaternion) -> tuple:
        euler =  euler_from_quaternion([
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w
        ])
        return Vector3(x=euler[0], y=euler[1], z=euler[2])
        
    def __init__(self, controller_type = Controller.OIAC, trajectory_type = Derivative.JERK):
        super().__init__('drone_feedback_controller')
        self.reset()
        self._init_controller(controller_type)
        self._init_derivative(trajectory_type)
        self._init_pubsub()
        self._init_gui()
        self._init_state()

    def _init_state(self):
        self.state = TrajectoryState.START

    def _init_derivative(self, derivative_type: Derivative):
        self.derivative_type = derivative_type

    def _init_gui(self):
        self.app = QApplication([])
        self.gui = App(self)
        self.gui.show()

    def _init_controller(self, controller_type: Controller):
        self.controller_type = controller_type
        match controller_type:
            case Controller.OIAC:
                self._init_OIAC_controller()
            case Controller.PID:
                self._init_PID_controller()

    def _init_OIAC_controller(self):
        self.throttle = OIAC(10, 1, 10)
        self.throttle.set_feedforward([5.45])
        self.throttle.set_saturation(-1, 2)
        print("OIAC controller")
        self.outer_loop = OIAC(360, 0.2, 8)
        self.outer_loop.set_saturation(-1, 1)
        self.inner_loop = OIAC(5, 0.6, 0.5)
        self.inner_loop.set_saturation(-0.01, 0.01)

    def _init_PID_controller(self):
        self.throttle = PID(2, 0.01, 0.1)
        self.throttle.set_feedforward([5.45])
        self.throttle.set_saturation(-1, 2)
        print("PID controller")
        self.outer_loop = PID(0.025, 0.0, 1)
        self.outer_loop.set_saturation(-1, 1)
        self.inner_loop = PID(0.005, 0, 1)
        self.inner_loop.set_saturation(-1, 1)

    def _init_pubsub(self):
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/Current_position',
            self.pose_callback,
            10)
        self.velocity_subscriber = self.create_subscription(
            TwistStamped,
            '/Current_velocity',
            self.velocity_callback,
            10)
        self.local_attitude_subscriber = self.create_subscription(
            Vector3(),
            '/Current_local_attitude',
            self.local_attitude_callback,
            10)
        self.thrust_publisher = self.create_publisher(Float64, '/Thrust', 10)
        self.outer_loop_publisher = self.create_publisher(Vector3, '/Attitude', 10)
        self.simulation_publisher_start = self.create_publisher(Bool, '/startSimulation', 10)
        self.simulation_publisher_stop = self.create_publisher(Bool, '/stopSimulation', 10)
        self.control_timing = self.create_timer(0.05, self.controller_callback)
        
        
    def reset(self):
        self._vx = 0.0
        self._vy = 0.0
        self._position = Point()
        self._orientation = Vector3()
        self._velocity = Vector3()
        self._angular_velocity = Vector3()
        self._thrust = Float64()
        self._torque = Vector3()
        self._rp = [0.0, 0.0, 0.0]
        self._twist = [0.0, 0.0, 0.0]
        self.__trajectory = np.array([[0.0, 0.0, 0.0]])
        self._max_timestep = 1
        self._cnt = 0
        self._count_up = True
        self._pos_data = []
        self._force_data = []
        self._K_values = []
        self._D_values = []
        self.__finished = False
        self._enable_data_storage = False
        self._use_min_dist = False
        self._use_trajectory = False
        self._min_dist = 0.5
        

    def controller_callback(self):
        match self.state:
            case TrajectoryState.START:
                self.start_simulation()
            case TrajectoryState.RUNNING:
                self.fly_drone()
                self.update_step()
                self.check_final_step()
            case TrajectoryState.FINAL:
                self.fly_drone()
                self.update_final_step()
            case TrajectoryState.STOP:
                self.__finished = True

    def update_final_step(self):
        self.append_positions_and_data()
        if (
                    self._use_min_dist
                    and self.is_under_min_dist(self._min_dist)
                    or not self._use_min_dist
                ):
            self.save_drone_path_data()

    def update_step(self):
        if self._use_min_dist:
            if self.is_under_min_dist(self._min_dist):
                self.append_positions_and_data()
                self._cnt += 1
        else:
            self._cnt += 0

    def should_store_data(self, enable):
        self._enable_data_storage = enable
        
    def should_use_min_dist(self, enable: bool, min_dist=0.1):
        self._use_min_dist = enable
        if enable:
            self._min_dist = min_dist
        
    def should_use_trajectory(self, enable):
        self._use_trajectory = enable
        if enable:
            self.import_trajectory(f'rrt_trajectory_{self.derivative_type.name}.csv')
        else:
            self.__trajectory = np.array([[-0.0, -0.0, 2.0]])
            
    
    def fly_drone(self):
        self.run_controller()
        self.assign_torque_values()
        self.check_out_of_control()
        self.publish_data()

    def append_positions_and_data(self):
        if not self._enable_data_storage:
            return
        self._pos_data.append([self._position.x, self._position.y, self._position.z])
        self._force_data.append([self._torque.x, self._torque.y, self._torque.z, self._thrust.data])
        if self.controller_type == Controller.OIAC:
            self._K_values.append(np.concatenate(([self.throttle.get_K()], self.outer_loop.get_K().ravel(), self.inner_loop.get_K().ravel())))
            self._D_values.append(np.concatenate(([self.throttle.get_D()], self.outer_loop.get_D().ravel(), self.inner_loop.get_D().ravel())))


    def is_under_min_dist(self, threshold):
        return np.linalg.norm(np.array(self.__trajectory[self._cnt]) - np.array([self._position.x, self._position.y, self._position.z])) < threshold

    def publish_data(self):
        self.app.processEvents()
        self.thrust_publisher.publish(self._thrust)
        self.outer_loop_publisher.publish(self._torque)

    def assign_torque_values(self):
        self._torque.x = -self._twist[0]
        self._torque.y = self._twist[1]
        self._torque.z = -self._twist[2]

    def run_controller(self):
        match self.controller_type:
            case Controller.OIAC:
                self.run_OIAC_controller()
            case Controller.PID:
                self.run_PID_controller()

    def run_OIAC_controller(self):
        self._rp = self.outer_loop.run_OIAC([self._position.x, self._position.y], self.__trajectory[self._cnt, :2], [self._velocity.x, self._velocity.y], [0.0, 0.0]).ravel()
        self._twist = self.inner_loop.run_OIAC([self._orientation.x, self._orientation.y, self._orientation.z], [-self._rp[1], self._rp[0], 0.0], [self._angular_velocity.x, self._angular_velocity.y, self._angular_velocity.z], [0.0, 0.0, 0.0])
        self._thrust.data = self.throttle.run_OIAC(self._position.z, self.__trajectory[self._cnt, 2], self._velocity.z, 0.0).ravel()[0]

    def run_PID_controller(self):
        self._rp = self.outer_loop.run_PID([self._position.x, self._position.y], self.__trajectory[self._cnt, :2], [-0.1*self._velocity.x, -0.1*self._velocity.y])
        self._twist = self.inner_loop.run_PID([self._orientation.x, self._orientation.y, self._orientation.z], [-self._rp[1], self._rp[0], 0.0])
        self._thrust.data = self.throttle.run_PID(self._position.z, self.__trajectory[self._cnt, 2], -2 * self._velocity.z).ravel()[0]

    def check_final_step(self):
        if self._cnt == self._max_timestep:
            self._cnt = self._max_timestep - 1
            self.state = TrajectoryState.FINAL
        
    def save_drone_path_data(self):
        if self._enable_data_storage:
            np.savetxt(f"./src/drone_controller_vrep/resource/data/drone_path_{self.derivative_type.name}_{self.controller_type.name}.csv", self._pos_data, delimiter=",")
            np.savetxt(f"./src/drone_controller_vrep/resource/data/drone_force_and_torque_{self.derivative_type.name}_{self.controller_type.name}.csv", self._force_data, delimiter=",")
            if self.controller_type == Controller.OIAC:
                np.savetxt(f"./src/drone_controller_vrep/resource/data/drone_K_values_{self.derivative_type.name}.csv", self._K_values, delimiter=",")
                np.savetxt(f"./src/drone_controller_vrep/resource/data/drone_D_values_{self.derivative_type.name}.csv", self._D_values, delimiter=",")
        self.state = TrajectoryState.STOP

    def pose_callback(self, msg):
        self._position = msg.pose.position
        self._orientation = self.quaternion_to_euler(msg.pose.orientation)

    def velocity_callback(self, msg):
        self._velocity = msg.twist.linear
        self._angular_velocity = msg.twist.angular

    def local_attitude_callback(self, msg):
        self._vx = msg.x - msg.z
        self._vy = msg.y - msg.z

    def start_simulation(self):
        self.simulation_publisher_start.publish(Bool(data=True))
        self.state = TrajectoryState.RUNNING
        self.__finished = False

    def check_out_of_control(self):
        if self._position.x < -3 or self._position.x > 3 or self._position.y < -3 or self._position.y > 3 or self._position.z < 0 or self._position.z > 20 or self._orientation.x < -1.1 or self._orientation.x > 1.1 or self._orientation.y < -1.1 or self._orientation.y > 1.1:
            raise Exception("Dronee is gone out of control!!!!")
    
    def stop_simulation(self):
        print("Stopping")
        self.simulation_publisher_start.publish(Bool(data=False))
        self.simulation_publisher_stop.publish(Bool(data=True))
        self.gui.close()
        self.app.quit()

    def get_data(self):
        return {
            "X position": self._position.x,
            "phi target": self._rp[0],
            "phi current": self._orientation.y,
            "phi dot": self._angular_velocity.y,
            "roll torque": self._twist[1],
            "Y position": self._position.y,
            "theta target": -self._rp[1],
            "theta current": self._orientation.x,
            "theta dot": self._angular_velocity.x,
            "pitch torque": -self._twist[0],
            "Z position": self._position.z,
            "X target": self.__trajectory[self._cnt, 0],
            "Y target": self.__trajectory[self._cnt, 1],
            "Path count": self._cnt,
            "psi dot": self._angular_velocity.z,
        }

    def import_trajectory(self, file):
        self.__trajectory = read_data(file)[:, :3]
        self._max_timestep = self.__trajectory.shape[0]

    def is_finished(self):
        return self.__finished

def main(args=None):
    rclpy.init(args=args)
    try:
        drone_controller = DroneFeedbackController(controller_type=Controller.OIAC, trajectory_type=Derivative.JERK)
        drone_controller.should_store_data(False)
        drone_controller.should_use_min_dist(False)
        drone_controller.should_use_trajectory(True)
        while rclpy.ok() and not drone_controller.is_finished():
            rclpy.spin_once(drone_controller, timeout_sec=0.05)
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        drone_controller.stop_simulation()
        drone_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()