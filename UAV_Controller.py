import numpy as np
import matplotlib.pyplot as plt

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
        
        # 初始状态
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.zeros(3)  # 欧拉角（roll, pitch, yaw）
        self.angular_velocity = np.zeros(3)
        
        # PID控制器参数
        # 设置P控制，ID控制为0
        self.Kp_inner = np.array([0, 0.1, 0])
        self.Ki_inner = np.array([0, 0, 0])
        self.Kd_inner = np.array([0, 0, 0])
        
        self.Kp_outer = np.array([0, 10, 0])
        self.Ki_outer = np.array([0, 0.0, 0])
        self.Kd_outer = np.array([0, 0.0, 0])
        
        # PID控制器误差积分项
        self.error_integral_inner = np.zeros(3)
        self.last_error_inner = np.zeros(3)
        
        self.error_integral_outer = np.zeros(3)
        self.last_error_outer = np.zeros(3)
        
        # 保存历史数据
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.orientation_history = []
        self.angular_velocity_history = []
        self.error_history = []
    
    def update(self, dt, target_orientation):
        # 姿态环
        # 外环PID控制
        outer_error = target_orientation - self.orientation
        
        self.error_integral_outer += outer_error * dt
        
        outer_error_derivative = (outer_error - self.last_error_outer) / dt
        
        inner_control_input = self.Kp_outer * outer_error + self.Ki_outer * self.error_integral_outer + self.Kd_outer * outer_error_derivative
        
        self.last_error_outer = outer_error

        # 内环PID控制
        inner_error = inner_control_input - self.angular_velocity
        
        self.error_integral_inner += inner_error * dt
        
        inner_error_derivative = (inner_error - self.last_error_inner) / dt
        
        inner_control_output = self.Kp_inner * inner_error + self.Ki_inner * self.error_integral_inner + self.Kd_inner * inner_error_derivative
        
        self.last_error_inner = inner_error

        # 位置环
        # # 内环PID控制
        # inner_error = target_orientation - self.angular_velocity
        
        # self.error_integral_inner += inner_error * dt
        
        # inner_error_derivative = (inner_error - self.last_error_inner) / dt
        
        # inner_control_input = self.Kp_inner * inner_error + self.Ki_inner * self.error_integral_inner + self.Kd_inner * inner_error_derivative
        
        # self.last_error_inner = inner_error
        
        # # 外环PID控制
        # outer_error = inner_control_input - self.orientation
        
        # self.error_integral_outer += outer_error * dt
        
        # outer_error_derivative = (outer_error - self.last_error_outer) / dt
        
        # outer_control_input = self.Kp_outer * outer_error + self.Ki_outer * self.error_integral_outer + self.Kd_outer * outer_error_derivative
        
        # self.last_error_outer = outer_error
        
        # 计算旋翼力和力矩
        forces = np.zeros(4)
        moments = np.zeros(3)
        
        forces[0] = self.mass * 9.8  # 垂直向上的重力
        
        moments[0] = inner_control_output[0]  # 绕X轴的力矩
        moments[1] = inner_control_output[1]  # 绕Y轴的力矩
        moments[2] = inner_control_output[2]  # 绕Z轴的力矩
        
        # 更新无人机状态
        self.position += self.velocity * dt
        self.velocity += forces[:3] / self.mass * dt
        self.orientation += self.angular_velocity * dt
        self.angular_velocity += np.linalg.inv(self.inertia) @ (moments - np.cross(self.angular_velocity, self.inertia @ self.angular_velocity)) * dt
        
        # 保存历史数据
        self.time_history.append(dt * len(self.time_history))
        self.orientation_history.append(self.orientation.tolist())
        self.angular_velocity_history.append(self.angular_velocity.tolist())
        self.velocity_history.append(self.velocity.tolist())
        self.position_history.append(self.position.tolist())
        self.error_history.append(outer_error)
    
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
    target_orientation = np.array([0, 0.3, 0])  # 目标姿态角
    quadcopter = Quadcopter()
    
    for i in range(2000):
        quadcopter.update(dt, target_orientation)
    
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

    plt.subplot(2,2,3)
    plt.plot(time_history, velocity_history)
    plt.xlabel('仿真时间')
    plt.ylabel('速度')
    plt.legend(['Vx', 'Vy', 'Vz'])

    plt.subplot(2,2,4)
    plt.plot(time_history, position_history)
    plt.xlabel('仿真时间')
    plt.ylabel('位置')
    plt.legend(['X', 'Y', 'Z'])
    plt.show()
