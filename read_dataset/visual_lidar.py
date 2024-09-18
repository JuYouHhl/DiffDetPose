import yaml
import matplotlib.pyplot as plt
import numpy as np

# 读取 YAML 文件
with open('F:/Rope3d/training/extrinsics/1632_fa2sd4a11North151_420_1613710840_1613716786_1_obstacle.yaml', 'r') as file:
    data = yaml.safe_load(file)
    
# 提取点云数据和变换矩阵
points_ground_coffe = data['points_ground_coffe']
transform = data['transform']

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q['w'],q['x'],q['y'],q['z']
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
R = quaternion_to_rotation_matrix(transform['rotation'])
T_vector = transform['translation']
T = np.array((T_vector['x'],T_vector['y'],T_vector['z']))

# 将点云数据转换为 NumPy 数组
points_ground_coffe = np.array(points_ground_coffe).reshape(-1, 3)

# 应用变换矩阵
# 注意：这里假设 transform 包含了从相机坐标系到世界坐标系的变换
# 并且点云数据是按照 XYZ 的顺序排列的
world_points = np.dot(points_ground_coffe, R) + T

# 可视化点云数据
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c=world_points[:, 2], cmap='viridis', s=1)
# ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2], c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 定义激光雷达坐标系的原点和轴向点
origin = np.zeros((1, 3))
# axis_lengths = [1, 1, 1]  # 轴的长度可以根据需要调整
# 应用变换矩阵到原点和轴向点
# transformed_origin = origin + T
# transformed_axis_x = origin + R.dot(np.array(axis_lengths) * [1, 0, 0])
# transformed_axis_y = origin + R.dot(np.array(axis_lengths) * [0, 1, 0])
# transformed_axis_z = origin + R.dot(np.array(axis_lengths) * [0, 0, 1])
# 绘制坐标轴
# A=transformed_origin.reshape(3) 
# B=transformed_axis_x.reshape(3) 
# plt.quiver(A[0], A[1], A[2], B[0]-A[0], B[1]-A[1], B[2]-A[2], color='r', arrow_length_ratio=0.5)

# 启用鼠标滚轮缩放功能
ax.view_init(elev=20, azim=30)
ax.dist = 8
plt.show()