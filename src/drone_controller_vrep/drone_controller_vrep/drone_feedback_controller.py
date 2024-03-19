import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped, Vector3, Quaternion, Point
from std_msgs.msg import Float64, Bool
from tf_transformations import euler_from_quaternion
from .online_impedance_adaptive_controller import OIAC
from .propotional_integral_derivative_controller import PID
import numpy as np
from .gui import App, QApplication
from enum import Enum

class Controller(Enum):
    OIAC = 0
    PID = 1

class TrajectoryState(Enum):
    START = 0
    RUNNING = 1
    FINAL = 2
    STOP = 3

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
        
    def __init__(self, controller_type = Controller.OIAC):
        super().__init__('drone_feedback_controller')
        self.reset()
        self._init_controller(controller_type)
        self._init_pubsub()
        self._init_gui()
        self._init_state()

    def _init_state(self):
        self.state = TrajectoryState.START

    def _init_gui(self):
        self.app = QApplication([])
        self.gui = App(self)
        self.gui.show()

    def _init_controller(self, controller_type):
        self.controller_type = controller_type
        match controller_type:
            case Controller.OIAC:
                self._init_OIAC_controller()
            case Controller.PID:
                self._init_PID_controller()

    def _init_OIAC_controller(self):
        self.throttle = OIAC(8, 0.5, 10)
        self.throttle.set_feedforward([5.45])
        self.throttle.set_saturation(-1, 2)
        print("OIAC controller")
        self.outer_loop = OIAC(360, 0.2, 8)
        self.outer_loop.set_saturation(-1, 1)
        self.inner_loop = OIAC(5, 0.6, 0.5)
        self.inner_loop.set_saturation(-0.01, 0.01)

    def _init_PID_controller(self):
        self.throttle = PID(2, 0, 0)
        self.throttle.set_feedforward([5.45])
        self.throttle.set_saturation(-1, 2)
        print("PID controller")
        self.outer_loop = PID(0.025, 0, 0.0)
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
        self.vx = 0.0
        self.vy = 0.0
        self.position = Point()
        self.orientation = Vector3()
        self.velocity = Vector3()
        self.angular_velocity = Vector3()
        self.thrust = Float64()
        self.torque = Vector3()
        self.rp = [0.0, 0.0, 0.0]
        self.twist = [0.0, 0.0, 0.0]
        self.trajectory = np.array([[0.0, 0.0]])
        self.max_timestep = 1
        self.cnt = 0
        self.count_up = True
        self.pos_data = np.empty((0, 3))
        self.force_data = np.empty((0, 4))
        self.finished = False
        

    def controller_callback(self):
        match self.state:
            case TrajectoryState.START:
                self.start_simulation()
            case TrajectoryState.RUNNING:
                self.fly_drone()
                if self.is_under_min_dist(0.5):
                    self.append_positions_and_torque()
                    self.cnt += 1
                self.check_final_step()
            case TrajectoryState.FINAL:
                self.fly_drone()
                self.append_positions_and_torque()
                if self.is_under_min_dist(0.1):
                    self.save_drone_path_data()
            case TrajectoryState.STOP:
                self.finished = True

    def fly_drone(self):
        self.run_controller()
        self.assign_torque_values()
        self.check_out_of_control()
        self.publish_data()

    def append_positions_and_torque(self):
        self.pos_data = np.append(self.pos_data, np.array([[self.position.x, self.position.y, self.position.z]]), axis=0)
        self.force_data = np.append(self.force_data, np.array([[self.torque.x, self.torque.y, self.torque.z, self.thrust.data]]), axis=0)

    def is_under_min_dist(self, threshold):
        return np.linalg.norm(np.array(self.trajectory[self.cnt]) - np.array([self.position.x, self.position.y])) < threshold

    def publish_data(self):
        self.app.processEvents()
        self.thrust_publisher.publish(self.thrust)
        self.outer_loop_publisher.publish(self.torque)

    def assign_torque_values(self):
        self.torque.x = -self.twist[0]
        self.torque.y = self.twist[1]
        self.torque.z = -self.twist[2]

    def run_controller(self):
        match self.controller_type:
            case Controller.OIAC:
                self.run_OIAC_controller()
            case Controller.PID:
                self.run_PID_controller()

    def run_OIAC_controller(self):
        self.rp = self.outer_loop.run_OIAC([self.position.x, self.position.y], self.trajectory[self.cnt], [self.velocity.x, self.velocity.y], [0.0, 0.0]).ravel()
        self.twist = self.inner_loop.run_OIAC([self.orientation.x, self.orientation.y, self.orientation.z], [-self.rp[1], self.rp[0], 0.0], [self.angular_velocity.x, self.angular_velocity.y, self.angular_velocity.z], [0.0, 0.0, 0.0])
        self.thrust.data = self.throttle.run_OIAC(self.position.z, 1.0, self.velocity.z, 0.0).ravel()[0]

    def run_PID_controller(self):
        self.rp = self.outer_loop.run_PID([self.position.x, self.position.y], self.trajectory[self.cnt], [-0.1*self.velocity.x, -0.1*self.velocity.y])
        self.twist = self.inner_loop.run_PID([self.orientation.x, self.orientation.y, self.orientation.z], [-self.rp[1], self.rp[0], 0.0])
        self.thrust.data = self.throttle.run_PID(self.position.z, 1.0, -2 * self.velocity.z).ravel()[0]

    def check_final_step(self):
        if self.cnt == self.max_timestep:
            self.cnt = self.max_timestep - 1
            self.state = TrajectoryState.FINAL
        
    def save_drone_path_data(self):    
        np.savetxt("./src/drone_controller_vrep/resource/data/drone_path_straight.csv", self.pos_data, delimiter=",")
        np.savetxt("./src/drone_controller_vrep/resource/data/drone_force_and_torque_straight.csv", self.force_data, delimiter=",")
        self.state = TrajectoryState.STOP

    def pose_callback(self, msg):
        self.position = msg.pose.position
        self.orientation = self.quaternion_to_euler(msg.pose.orientation)

    def velocity_callback(self, msg):
        self.velocity = msg.twist.linear
        self.angular_velocity = msg.twist.angular

    def local_attitude_callback(self, msg):
        self.vx = msg.x - msg.z
        self.vy = msg.y - msg.z

    def start_simulation(self):
        self.simulation_publisher_start.publish(Bool(data=True))
        self.state = TrajectoryState.RUNNING
        self.finished = False

    def check_out_of_control(self):
        if self.position.x < -3 or self.position.x > 3 or self.position.y < -3 or self.position.y > 3 or self.position.z < 0 or self.position.z > 20 or self.orientation.x < -1.1 or self.orientation.x > 1.1 or self.orientation.y < -1.1 or self.orientation.y > 1.1:
            raise Exception("Dronee is gone out of control!!!!")
    
    def stop_simulation(self):
        self.simulation_publisher_stop.publish(Bool(data=True))
        self.gui.close()
        self.app.quit()

    def get_data(self):
        return {
            "X position": self.position.x,
            "phi target": self.rp[0],
            "phi current": self.orientation.y,
            "phi dot": self.angular_velocity.y,
            "roll torque": self.twist[1],
            "Y position": self.position.y,
            "theta target": -self.rp[1],
            "theta current": self.orientation.x,
            "theta dot": self.angular_velocity.x,
            "pitch torque": -self.twist[0],
            "X target": self.trajectory[self.cnt, 0],
            "Y target": self.trajectory[self.cnt, 1],
            "Path count": self.cnt,
            "psi dot": self.angular_velocity.z,
            "yaw torque": -self.twist[2],
        }

    def import_trajectory(self, file):
        self.trajectory = np.loadtxt(file, delimiter=',')
        self.trajectory = self.trajectory[:, :2]
        self.max_timestep = self.trajectory.shape[0]

def main(args=None):
    rclpy.init(args=args)
    try:
        drone_controller = DroneFeedbackController(controller_type=Controller.PID)
        drone_controller.import_trajectory('./src/drone_controller_vrep/resource/data/straight_trajectory.csv')
        while rclpy.ok() and not drone_controller.finished:
            rclpy.spin_once(drone_controller, timeout_sec=0.05)
    except Exception as e:
        print(e)
    finally:
        drone_controller.stop_simulation()
        drone_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()