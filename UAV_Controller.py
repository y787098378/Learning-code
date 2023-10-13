import numpy as np
import matplotlib.pyplot as plt
import math

# 解决matplotlib绘图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.sans-serif'] = ['KaiTi']   # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False   # 解决保存图像是负号'-'显示为方块的问题

class Quadcopter:
    def __init__(self):
        # 无人机质量
        self.mass = 0.5
        
        # 惯性矩阵
        self.inertia = np.array([
            [0.0023, 0, 0],
            [0, 0.0023, 0],
            [0, 0, 0.004]
        ])
        
        # 旋翼力和力矩常数  
        self.kf = 6.11e-8
        self.km = 1.5e-9

        # X型四旋翼无人机 控制分配 添系数
        self.M = np.array([[1, 1.4142, 1.4142, 1],
                   [1, -1.4142, 1.4142, -1],
                   [1, -1.4142, -1.4142, 1],
                   [1, 1.4142, -1.4142, -1]])
        self.M[:, 0] *= self.kf
        self.M[:, 1:] *= self.km

        # 初始状态
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # 欧拉角（roll, pitch, yaw）
        self.angular_velocity = np.zeros(3)
        
        # PID控制器参数——————姿态环
        # 设置P控制，ID控制为0
        self.Kp_inner_euler = np.array([0, 0.1, 0])
        self.Ki_inner_euler = np.array([0, 0, 0])
        self.Kd_inner_euler = np.array([0, 0, 0])
        
        self.Kp_outer_euler = np.array([0, 10, 0])
        self.Ki_outer_euler = np.array([0, 0.0, 0])
        self.Kd_outer_euler = np.array([0, 0.0, 0])
        
        # PID控制器误差积分项
        self.error_integral_inner_euler = np.zeros(3)
        self.last_error_inner_euler = np.zeros(3)
        
        self.error_integral_outer_euler = np.zeros(3)
        self.last_error_outer_euler = np.zeros(3)


        # PID控制器参数——————位置环
        # 设置P控制，ID控制为0
        self.Kp_inner_position = np.array([0, 0.1, 0])
        self.Ki_inner_position = np.array([0, 0, 0])
        self.Kd_inner_position = np.array([0, 0, 0])
        
        self.Kp_outer_position = np.array([0, 1, 0])
        self.Ki_outer_position = np.array([0, 0.0, 0])
        self.Kd_outer_position = np.array([0, 0.0, 0])
        
        # PID控制器误差积分项
        self.error_integral_inner_position = np.zeros(3)
        self.last_error_inner_position = np.zeros(3)
        
        self.error_integral_outer_position = np.zeros(3)
        self.last_error_outer_position = np.zeros(3)
        
        # 保存历史数据
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.orientation_history = []
        self.angular_velocity_history = []
        self.error_history = []
        self.omege_sqare = []

    def update(self, dt, target_position):
        # 位置环
        # 外环PID控制
        outer_position_error = target_position - self.position
        
        self.error_integral_outer_position += outer_position_error * dt
        
        outer_position_error_derivative = (outer_position_error - self.last_error_outer_position) / dt
        
        inner_position_control_input = self.Kp_outer_position * outer_position_error + self.Ki_outer_position * self.error_integral_outer_position + self.Kd_outer_position * outer_position_error_derivative
        
        self.last_error_outer_position = outer_position_error

        # 内环PID控制
        inner_position_error = inner_position_control_input - self.velocity
        
        self.error_integral_inner_position += inner_position_error * dt
        
        inner_position_error_derivative = (inner_position_error - self.last_error_inner_position) / dt
        
        target_orientation = self.Kp_inner_position * inner_position_error + self.Ki_inner_position * self.error_integral_inner_position + self.Kd_inner_position * inner_position_error_derivative
        
        self.last_error_inner_position = inner_position_error

        # 姿态环
        # 外环PID控制
        outer_euler_error = target_orientation - self.orientation
        
        self.error_integral_outer_euler += outer_euler_error * dt
        
        outer_euler_error_derivative = (outer_euler_error - self.last_error_outer_euler) / dt
        
        inner_euler_control_input = self.Kp_outer_euler * outer_euler_error + self.Ki_outer_euler * self.error_integral_outer_euler + self.Kd_outer_euler * outer_euler_error_derivative
        
        self.last_error_outer_euler = outer_euler_error

        # 内环PID控制
        inner_euler_error = inner_euler_control_input - self.angular_velocity
        
        self.error_integral_inner_euler += inner_euler_error * dt
        
        inner_euler_error_derivative = (inner_euler_error - self.last_error_inner_euler) / dt
        
        inner_euler_control_output = self.Kp_inner_euler * inner_euler_error + self.Ki_inner_euler * self.error_integral_inner_euler + self.Kd_inner_euler * inner_euler_error_derivative
        
        self.last_error_inner_euler = inner_euler_error


        # 位置环
        # # 内环PID控制
        # inner_euler_error = target_orientation - self.angular_velocity
        
        # self.error_integral_inner_euler += inner_euler_error * dt
        
        # inner_euler_error_derivative = (inner_euler_error - self.last_error_inner_euler) / dt
        
        # inner_euler_control_input = self.Kp_inner_euler * inner_euler_error + self.Ki_inner_euler * self.error_integral_inner_euler + self.Kd_inner_euler * inner_euler_error_derivative
        
        # self.last_error_inner_euler = inner_euler_error
        
        # # 外环PID控制
        # outer_euler_error = inner_euler_control_input - self.orientation
        
        # self.error_integral_outer_euler += outer_euler_error * dt
        
        # outer_euler_error_derivative = (outer_euler_error - self.last_error_outer_euler) / dt
        
        # outer_euler_control_input = self.Kp_outer_euler * outer_euler_error + self.Ki_outer_euler * self.error_integral_outer_euler + self.Kd_outer_euler * outer_euler_error_derivative
        
        # self.last_error_outer_euler = outer_euler_error
        
        # 计算旋翼力和力矩
        forces = np.zeros(4)
        moments = np.zeros(3)

        moments[0] = inner_euler_control_output[0]  # 绕X轴的力矩
        moments[1] = inner_euler_control_output[1]  # 绕Y轴的力矩
        moments[2] = inner_euler_control_output[2]  # 绕Z轴的力矩
        
        #根据力矩需要解算力
        #total_force = np.sum(force)
        #total_force = self.mass * np.array([0, 0, 9.8])   # 垂直向上的重力
        total_force = np.array([[self.mass * 9.8/math.cos(self.orientation[1])]])
        
        # 将向量和数字矩阵垂直拼接
        Desire_FM = np.vstack((total_force, moments .reshape(-1, 1)))
        omega_square =  np.dot(self.M,Desire_FM ) 
        #print(omega_square)

        # 更新无人机状态
        
        self.position += self.velocity * dt
        self.velocity += total_force[0]*math.tan(self.orientation[1]) / self.mass * dt
        self.orientation += self.angular_velocity * dt
        self.angular_velocity += np.linalg.inv(self.inertia) @ (moments - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)) * dt
        
        # 保存历史数据
        self.time_history.append(dt * len(self.time_history))
        self.orientation_history.append(self.orientation.tolist())
        self.angular_velocity_history.append(self.angular_velocity.tolist())
        self.velocity_history.append(self.velocity.tolist())
        self.position_history.append(self.position.tolist())
        self.error_history.append(outer_euler_error)
        self.omege_sqare.append(omega_square)

    def set_position(self, position):
        self.position = position
    
    def set_velocity(self, velocity):
        self.velocity = velocity
    
    def set_orientation(self, orientation):
        self.orientation = orientation
    
    def set_angular_velocity(self, angular_velocity):
        self.angular_velocity = angular_velocity

# 测试代码
if __name__ == "__main__":
    dt = 0.005  # 时间步长
    #target_orientation = np.array([0, 0.3, 0])  # 目标姿态角
    target_position = np.array([0, 10, 0])  # 目标位置（X=10，y=0）
    quadcopter = Quadcopter()
    
    for i in range(1000):
        #quadcopter.update(dt, target_orientation)
        quadcopter.update(dt, target_position)
    # 将历史数据绘制成图形
    time_history = quadcopter.time_history
    orientation_history = quadcopter.orientation_history
    angular_velocity_history = quadcopter.angular_velocity_history
    velocity_history = quadcopter.velocity_history
    position_history = quadcopter.position_history
    # 绘制子图
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(time_history, angular_velocity_history)
    plt.xlabel('仿真时间')
    plt.ylabel('角速率')
    plt.legend(['p', 'q', 'r'])

    plt.subplot(2,2,2)
    plt.plot(time_history, orientation_history)
    plt.xlabel('仿真时间')
    plt.ylabel('姿态角')
    plt.legend(['Roll', 'Pitch', 'Yaw'])

    # plt.subplot(2,2,3)
    # plt.plot(time_history, velocity_history)
    # plt.xlabel('仿真时间')
    # plt.ylabel('速度')
    # plt.legend(['Vx', 'Vy', 'Vz'])

    # plt.subplot(2,2,4)
    # plt.plot(time_history, position_history)
    # plt.xlabel('仿真时间')
    # plt.ylabel('位置')
    # plt.legend(['X', 'Y', 'Z'])
    # plt.show()

    plt.subplot(2,2,3)
    plt.plot(time_history, velocity_history)
    plt.xlabel('仿真时间')
    plt.ylabel('速度')
    plt.legend(['Vx'])

    plt.subplot(2,2,4)
    plt.plot(time_history, position_history)
    plt.xlabel('仿真时间')
    plt.ylabel('位置')
    plt.legend(['X'])
    plt.show()
