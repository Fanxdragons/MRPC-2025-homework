import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

#参数定义
OMEGA = 0.5            
ALPHA = np.pi / 12    
INPUT_FILE = '/home/stuwork/MRPC-2025-homework/documents/tracking.csv'  

def calculate_device_pose():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"错误：找不到 {INPUT_FILE}，请确认文件路径。")
        return
    results = []
    for index, row in df.iterrows():
        t = row['time'] if 'time' in row else row[0]      
        # --- 获取机体姿态 R_wb ---
        # Scipy Rotation 输入顺序默认是 [x, y, z, w]
        q_wb = [row['qx'], row['qy'], row['qz'], row['qw']] 
        r_wb = R.from_quat(q_wb).as_matrix()
        # --- 构建相对姿态 R_bd ---
        sw = np.sin(OMEGA * t)
        cw = np.cos(OMEGA * t)
        sa = np.sin(ALPHA)
        ca = np.cos(ALPHA)
        # 构造矩阵 
        r_bd = np.array([
            [cw,    -sw * ca,   sw * sa],
            [sw,     cw * ca,  -cw * sa],
            [0,      sa,        ca     ]
        ])

        # --- 坐标系变换 R_wd = R_wb * R_bd ---
        r_wd = np.matmul(r_wb, r_bd)
        # --- 转回四元数 ---
        quat_wd = R.from_matrix(r_wd).as_quat() # 返回 [x, y, z, w]
        # --- 强制 w >= 0 (连续性处理) ---
        if quat_wd[3] < 0:
            quat_wd = -quat_wd

        results.append([t, quat_wd[0], quat_wd[1], quat_wd[2], quat_wd[3]])
    res_df = pd.DataFrame(results, columns=['t', 'qx', 'qy', 'qz', 'qw'])
    return res_df

def plot_quaternion(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df['t'], df['qw'], label='w', linewidth=2)
    plt.plot(df['t'], df['qx'], label='x', linewidth=2)
    plt.plot(df['t'], df['qy'], label='y', linewidth=2)
    plt.plot(df['t'], df['qz'], label='z', linewidth=2)
    plt.title("End-Effector Orientation (Quaternion) in World Frame", fontsize=14)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Quaternion Value", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig("quaternion_plot.png", dpi=300)
    plt.show()
    print("绘图完成，已保存为 quaternion_plot.png")

if __name__ == "__main__":
    data = calculate_device_pose()
    if data is not None:
        plot_quaternion(data)
        # 打印前几行看看结果
        print(data.head())