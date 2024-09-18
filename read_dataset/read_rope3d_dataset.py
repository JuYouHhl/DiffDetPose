import os
import csv
import math
import random
import mmcv
import cv2
import numpy as np
# from pyquaternion import Quaternion
from tqdm import tqdm
import sys 
sys.path.append("./") 
from diffusiondet import DiffusionDetDatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog

name2nuscenceclass = {
    "car": "vehicle.car",
    "van": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus.rigid",
    "cyclist": "vehicle.bicycle",
    "tricyclist": "vehicle.trailer",
    "motorcyclist": "vehicle.motorcycle",
    "pedestrian": "human.pedestrian.adult",
    "trafficcone": "movable_object.trafficcone",
}
from PIL import Image

# 功能：查看图片能否打开，有没有这个图片、是不是完好、没有损坏的图片
# 参数：图片路径
# 返回：True
def check_pic(path_pic):
    try:
        img = Image.open(path_pic)    # 如果图片不存在，报错FileNotFoundError
        img.load()    # 如果图片不完整，报错OSError: image file is truncated
        return True
    except (FileNotFoundError, OSError):
        # print('文件损坏')
        return False
    
def alpha2roty(alpha, pos):
    ry = alpha + np.arctan2(pos[0], pos[2])
    if ry > np.pi:
        ry -= 2 * np.pi
    if ry < -np.pi:
        ry += 2 * np.pi
    return ry

def clip2pi(ry):
    if ry > 2 * np.pi:
        ry -= 2 * np.pi
    if ry < - 2* np.pi:
        ry += 2 * np.pi
    return ry

def load_calib(calib_file):
    with open(calib_file, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        for line, row in enumerate(reader):
            if row[0] == 'P2:':
                P2 = row[1:]
                P2 = [float(i) for i in P2]
                P2 = np.array(P2, dtype=np.float32).reshape(3, 4)
                continue
    return P2

def load_denorm(denorm_file):
    with open(denorm_file, 'r') as f:
        lines = f.readlines()
    denorm = np.array([float(item) for item in lines[0].split(' ')])
    return denorm

def get_annos(label_path):# Tr_cam2lidar变换矩阵
    """ 
    args：
        label_path:label路径。障碍物在像素坐标系上的二维标注和相机坐标系上的三维标注；kitti格式的标签
        Tr_cam2lidar:4*4的变换矩阵(相机->雷达) 
    return：
        annos:障碍物的标注信息
            dim                 3维物体的尺寸（长、宽、高）！！！需要
            loc                 相机坐标系下的物体位置lx,ly,lz
            rotation            旋转角度
            name                障碍物类别
            box2d               物体的2维边界框
            truncated_state     截断状态
            occluded_state      遮挡状态
    """
    fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',
                      'dl', 'lx', 'ly', 'lz', 'ry']
    annos = []
    with open(label_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)
        for line, row in enumerate(reader):
            if row["type"] in name2nuscenceclass.keys():
                name = name2nuscenceclass[row["type"]]
                box2d = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]# 二维边界框
                # 过滤点和线的错误的真值
                if box2d[0] >= box2d[2] or box2d[1] >= box2d[3]:
                    print("row：" + str(line))
                    print("label_file："+ label_path)
                    continue #立即跳出当次执行，进入下一次循环
                
                location = [float(row['lx']), float(row['ly']), float(row['lz'])]
                dim = [float(row['dl']), float(row['dw']), float(row['dh'])]# 三维物体的尺寸（长、宽、高）单位为米
                if dim[0]==0 or dim[1]==0 or dim[2]==0:
                    # print("lwh is false："+ str(dim))
                    continue # break 跳出并结束当前循环的执行
                
                alpha = float(row["alpha"])
                pos = np.array((float(row['lx']), float(row['ly']), float(row['lz'])), dtype=np.float32)# 原相机坐标系下三维物体的位置
                ry = float(row["ry"])
                if alpha > np.pi:
                    alpha -= 2 * np.pi
                    ry = alpha2roty(alpha, pos)
                rotation_y = ry # 与可视化做相同处理
                anno = {"name": name, "box2d": box2d, "location": location, "dim": dim, "rotation_y": rotation_y}
                annos.append(anno)

    return annos

def matrix2angle(bz_rotation_matrix):
    R = bz_rotation_matrix
    sy = np.sqrt(R[0, 0] * R[0, 0] +  R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    factor = (180.0 / math.pi)
    if not singular:
        rx = math.atan2(R[2, 1] , R[2, 2]) # 
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0
    # Convert radians to Euler angles
    angle = np.array((rx*factor,ry*factor,rz*factor), dtype=np.float32)
    return angle

def angle2matrix(bz_r_pose):
    # 将角度转换为弧度
    bz_r_pose_rad = bz_r_pose * (math.pi / 180.0)

    # 计算欧拉角的余弦和正弦值
    cos_x = np.cos(bz_r_pose_rad[0])
    sin_x = np.sin(bz_r_pose_rad[0])
    cos_y = np.cos(bz_r_pose_rad[1])
    sin_y = np.sin(bz_r_pose_rad[1])
    cos_z = np.cos(bz_r_pose_rad[2])
    sin_z = np.sin(bz_r_pose_rad[2])

    # 构建旋转矩阵
    rotation_matrixes = np.zeros((3, 3), dtype=np.float32)
    rotation_matrixes[0, 0] = cos_y * cos_z
    rotation_matrixes[0, 1] = - cos_x * sin_z + sin_x * sin_y * cos_z
    rotation_matrixes[0, 2] = sin_x * sin_z + cos_x * sin_y * cos_z
    rotation_matrixes[1, 0] = cos_y * sin_z
    rotation_matrixes[1, 1] = cos_x * cos_z + sin_x * sin_y * sin_z
    rotation_matrixes[1, 2] = - sin_x * cos_z + cos_x * sin_y * sin_z
    rotation_matrixes[2, 0] = - sin_y
    rotation_matrixes[2, 1] = sin_x * cos_y
    rotation_matrixes[2, 2] = cos_x * cos_y

    return rotation_matrixes

def box_3d(dim, location, rotation_matrix): # lwh,location,rotation_angle 
    """ Test whether the conversion between Euler Angle and rotation matrix
    Args:
        dim ([3]): lwh
        location ([3]):location
        rotation_matrix ([3,3]): rotation_angle
    Returns:
        corners_3d([8,3]): The 3D coordinates of the 3dbox in the camera coordinate system
    """
    rotation_angle = matrix2angle(rotation_matrix)
    l, w, h = dim[0], dim[1], dim[2]
    # 3d框在目标物体坐标系下的中心坐标为[0,-h/2,0]
    # 3d框的8个顶点在x、y、z轴上的坐标
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    R = angle2matrix(rotation_angle)
    corners_3d = np.dot(R, corners) + np.array(location, dtype=np.float32).reshape(3, 1)
    
    return corners_3d.transpose(1, 0)

def box_3d_origin(dim, location, rotation_y, denorm):
    """得到相机坐标系下的3D包围框
    https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb
    Args:
        dim (_type_):3维物体的尺寸（lwh）
        location (_type_):3维物体的位置（x、y、z）在相机坐标系下的表示
        rotation_y (_type_):在相机坐标系下，物体的全局方向角（物体前进方向与相机坐标系x轴的夹角）(rppe3d的解释)
        denorm (_type_): 地平面四个参数
    Returns:
        corners_3d
    """
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    # 目标物体坐标系的XOZ平面与地平面平行，旋转矩阵中Y轴分量不变；这个旋转矩阵将物体的前进方向旋转到相机坐标系的X轴上
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[0], dim[1], dim[2]
    # 3d框在目标物体坐标系下的中心坐标为[0,-h/2,0]
    # 3d框的8个顶点在x、y、z轴上的坐标
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) # 这个旋转矩阵将物体的前进方向旋转到相机坐标系的X轴上

    denorm = denorm[:3]
    denorm_norm = denorm / np.sqrt(denorm[0]**2 + denorm[1]**2 + denorm[2]**2)  # 平面单位法向量
    ori_denorm = np.array([0.0, -1.0, 0.0])
    theta = -1 * math.acos(np.dot(denorm_norm, ori_denorm))
    n_vector = np.cross(denorm, ori_denorm)
    n_vector_norm = n_vector / np.sqrt(n_vector[0]**2 + n_vector[1]**2 + n_vector[2]**2)
    rotation_matrix, j = cv2.Rodrigues(theta * n_vector_norm)
    rotation = np.dot(rotation_matrix, R)
    corners_3d = np.dot(rotation_matrix, corners_3d)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    
    return corners_3d.transpose(1, 0), rotation  # 得到相机坐标系下的3D包围框坐标

def generate_info_rope3d(rope3d_root, lenth=None, split='train'):
    """ info结构
            ——cam_info      相机参数
                ——height        图像高
                ——width         图像宽
                ——filename      图像绝对路径
                ... 
            ——ann_infos         标注参数
                ——ann_info              该张图像上某个目标物体的标注真值
                    ——category_name     3D目标类别
                    ——box2d             二维边界框(xyxy) [4]
                    ——location          3维物体的位置（x、y、z），单位为米
                    ——lwh               三维物体的尺寸（长、宽、高）
                    ——rotation_y        与可视化做同步处理的rotation_y
                    ——P2                相机内参3*4，即可视化中的P2
                    ——box3d             3dbox八个点的三维坐标（x、y、z）
                ——ann_info
                ...
        """
    # src_dir = os.path.join(rope3d_root, "training")
    # img_path = ["training-image_2a", "training-image_2b", "training-image_2c", "training-image_2d"]
    if split == 'train':
        src_dir = os.path.join(rope3d_root, "training")
        img_path = ["training-image_2a"]#, "training-image_2b", "training-image_2c", "training-image_2d"]
        # lenth=1000
    else:
        src_dir = os.path.join(rope3d_root, "validation")
        img_path = ["validation-image_2"]
        # lenth=428
    label_path = os.path.join(src_dir, "label_2")
    calib_path = os.path.join(src_dir, "calib")
    denorm_path = os.path.join(src_dir, "denorm")
    split_txt = os.path.join(src_dir, "train.txt" if split=='train' else 'val.txt')
    idx_list = [x.strip() for x in open(split_txt).readlines()]
    idx_list_valid = []
     
    infos = list()
    for index in idx_list:
        for sub_img_path in img_path:
            img_file = os.path.join(rope3d_root, sub_img_path, index + ".jpg")
            if os.path.exists(img_file): 
                idx_list_valid.append((sub_img_path, index))
                break
    gt_nums=0        
    for idx in tqdm(range(len(idx_list_valid))):
    # for idx in tqdm(range(3000)): # lenth 
        sub_img_path, index = idx_list_valid[idx]
        img_file = os.path.join(rope3d_root,sub_img_path, index + ".jpg")
        label_file = os.path.join(label_path, index + ".txt")
        calib_file = os.path.join(calib_path, index + ".txt")
        denorm_file = os.path.join(denorm_path, index + ".txt")
        
        if check_pic(img_file) == False:
            print("This jpg is truncated：" + img_file)
            continue
        
        info = dict()
        cam_info = dict()
        cam_info = dict()
        cam_info['height'] = 1080
        cam_info['width'] = 1920
        cam_info['filename'] = img_file 
        cam_info["imgId"] = index  # name id 
        cam_info["imgidx"] = idx # number id    

        P2 = load_calib(calib_file)# 相机内参矩阵3x4,第四列是补的0，返回 P2 的前3x3部分，即投影矩阵的左上角3x3矩阵
        denorm = load_denorm(denorm_file)# 地面方程
        cam_info['denorm'] = denorm 
        cam_info['P2'] = P2
      
        info['cam_info'] = cam_info
        ann_infos = list()
        annos = get_annos(label_file)
        if annos==[]:
            # print("False")
            continue
        gt_nums+=1
        for anno in annos:      
            ann_info = dict()
            ann_info["category_name"] = anno["name"]# 3D目标类别         
            ann_info["box2d"] = anno["box2d"] # 二维边界框(xyxy abs) [4]
            ann_info["lwh"] = anno["dim"] # 三维物体的尺寸（长、宽、高）
            ann_info["rotation_y"] = anno["rotation_y"] # 与可视化做同步处理的rotation_y
            ann_info["P2"] = info['cam_info']['P2'] # 相机内参3*4，即可视化中的P2
            # box3d：[8,3]；rotation：[3,3]
            ann_info["box3d"], rotation_matrix = box_3d_origin(anno["dim"],anno["location"],anno["rotation_y"],info['cam_info']['denorm']) # [8,3]
            rotation = matrix2angle(rotation_matrix)
            # box3d = box_3d(anno["dim"],anno["location"],rotation_matrix)
            ann_info["location"] = [anno["location"][0]/31,anno["location"][1]/21,anno["location"][2]/100] #3维物体的位置（x、y、z），单位为米；
            ann_info["rotation"] = np.array([rotation[0]/90,rotation[1]/40,rotation[2]/90,]) # (3,)
            ann_infos.append(ann_info)
        info['ann_infos'] = ann_infos
        infos.append(info)
    return infos

def projection(P, pts_3d):
    """将3dbox投影到图像上，并获取这八个点的最小外接矩形

    Args:
        bz_tgt_P2 (tensor[13,3,4]): 相机投影矩阵
        bz_src_3dboxes (tensor[13,8,3]): 需要投影的3dbox的八个点的三维坐标
    return:
        min_max_coords (tensor[13,4,2]):八个点的最小外接矩形的四个点的二维坐标
    """
    pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    return pts_2d   

def main():
    rope3d_root = "F:\Rope3D"
    train_infos = generate_info_rope3d(rope3d_root, split='train')
    val_infos = generate_info_rope3d(rope3d_root, split='val')

    total_infos = train_infos + val_infos
    random.shuffle(total_infos)
    train_infos = total_infos[:int(0.7 * len(total_infos))]
    val_infos = total_infos[int(0.7 * len(total_infos)):]
    # mmcv.dump(train_infos, './data/rope3d/rope3d_train_a.pkl')
    mmcv.dump(train_infos, './data/rope3d/rope3d_all_train.pkl') # 40320
    mmcv.dump(val_infos, './data/rope3d/rope3d_all_val.pkl') # 4666

if __name__ == '__main__':
    main()

# print_instances_class_histogram(dataset_dicts, class_names) 增加数据集类别分布直方图的绘制
