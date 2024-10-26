import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  
import numpy as np
import argparse  
  
def parse_arguments():  
    parser = argparse.ArgumentParser(description='Parameter Definitions')  
      
    parser.add_argument('--batch_num', type=int, default=100, help='Number of points in each batch of tracks')  
    parser.add_argument('--radar_length', type=int, default=20, help='Track length')  
    parser.add_argument('--r', type=float, default=0.9, help='Generate radar tracks by adding random errors with sigma r to AIS data')  
    parser.add_argument('--r_threshold', type=float, default=0.9, help='Sigma in the probability distribution function') 
    parser.add_argument('--threshold', type=float, default=0.4, help='Only scores exceeding the Threshold will be recorded')
    parser.add_argument('--resolution', type=float, default=0.01, help='Resolution of the parameter space for voting')  
      
    args = parser.parse_args()  
    return args  

np.random.seed(0)

# 定义生成S中点的函数，现在用的是正弦函数
def generate_point(x):
    y = np.sin(x)  
    return np.array([x, y])

# 带边界判定的类高斯函数，这是为了避免在边界外叠加出最大值，当杂波点极其多的时候可能会导致这个问题
def gaussian(x, y, r, mux=0, muy=0):
    distance_squared = (x - mux) ** 2 + (y - muy) ** 2
    mask_inside = distance_squared <= r**2
    mask_outside = distance_squared > r**2

    distance_squared = (x - mux) ** 2 + (y - muy) ** 2 - r**2
    Z_inside = -1 + np.exp(-distance_squared / (2 * r**2))
    # Z_inside = np.exp(-distance_squared) - np.exp(-(r**2))
    Z_outside = np.zeros_like(x, dtype=float)

    return np.where(mask_inside, Z_inside, Z_outside)


# ********************************************参数定义
args = parse_arguments()  
batch_num = args.batch_num  
radar_length = args.radar_length  
r = args.r  
r_threshold = args.r_threshold  
threshold = args.threshold  
resolution = args.resolution  

# ********************************************数据生成
# AIS数据
AIS_x = np.linspace(0, 0 + 2 * 3.14159, radar_length)
AIS = np.array([generate_point(x) for x in AIS_x])

# 生成雷达航迹航迹
mean = [0, 3] 
cov = [[0.09, 0], [0, 0.09]]  
delta_1 = np.random.multivariate_normal(mean, cov, 20)
radar_real_1 = AIS + delta_1

# 添加错误点迹
random_points_list = []
for i in range(radar_length):
    # 生成一个batch_nums*2的随机数组
    random_array = np.random.uniform(low=[-3, -3], high=[8, 8], size=(batch_num, 2))
    # 将最后一个点设置为radar_real中的对应点
    random_array[batch_num - 1, :] = radar_real_1[i, :]
    random_points_list.append(random_array)


# ********************************************生成数据可视化
# 生成M种不同的颜色
colors = plt.cm.rainbow(np.linspace(0, 1, radar_length))
# # 绘制所有点
# for i, array in enumerate(random_points_list):
#     plt.scatter(array[:, 0], array[:, 1], color=colors[i])
# 高亮AIS数据
plt.scatter(AIS[:, 0], AIS[:, 1], color="red", s=100, marker="x", label="AIS Data")
# 高亮航迹数据
plt.scatter(
    radar_real_1[:, 0],
    radar_real_1[:, 1],
    color="blue",
    s=100,
    marker="x",
    label="radar_line Data",
)
# # 给 AIS 数据点标注数字
# for i, (x, y) in enumerate(AIS):
#     plt.annotate(str(i+1), (x, y), textcoords="offset points", xytext=(5,5), ha='center')
# # 给 radar_real_1 数据点标注数字
# for i, (x, y) in enumerate(radar_real_1):
#     plt.annotate(str(i+1), (x, y), textcoords="offset points", xytext=(5,5), ha='center')
# 添加图表标题和标签
plt.title("simulated data")
plt.xlabel("longitude(km)")
plt.ylabel("latitude(km)")
plt.legend()
plt.grid(True)
plt.show()


# ********************************************差分变换
# 差分变换
difference_list = []
for i, array in enumerate(random_points_list):
    difference = array - AIS[i]
    difference_list.append(difference)

# ********************************************差分变换可视化
# 绘制所有差异点
for i, difference in enumerate(difference_list):
    plt.scatter(difference[:, 0], difference[:, 1], color=colors[i])

# 添加图表标题和标签
plt.title("Differential Transformation")
plt.xlabel("δlongitude(km)")
plt.ylabel("δlatitude(km)")
plt.grid(True)
plt.show()

# ********************************************点云生成
# 生成网格
x = np.arange(-4, 10, resolution)
y = np.arange(-4, 10, resolution)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X, dtype=float)

for mux, muy in difference_list[0]:
    Z += gaussian(X, Y, r, mux, muy)

# ********************************************点云生成可视化
plt.figure(figsize=(10, 6))
plt.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", vmin=0)
plt.colorbar()
plt.title("density estimation of the first batch of radar point data")
plt.xlabel("δlongitude(km)")
plt.ylabel("δlatitude(km)")
plt.show()

# ********************************************门限滤出
# 创建一个足够大的数组 K 来存储每个 DIFFERENCE_LIST 向量对应的 Z 矩阵
K = np.zeros((radar_length, len(y), len(x)))
for i, array in enumerate(difference_list):
    for mux, muy in array:
        K[i] += gaussian(X, Y, r_threshold, mux, muy)

# 使用滤波器检索 K，找出所有对应点都大于门限值的点
mask = np.all(K > threshold, axis=0)

# ********************************************门限滤出可视化
plt.figure(figsize=(10, 6))
plt.imshow(
    mask, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap="hot"
)
plt.title("Threshold="+str(threshold)+" ,r = "+str(r))
plt.xlabel("δlongitude(km)")
plt.ylabel("δlatitude(km)")
plt.show()
  
# 假设我们要将这些样本聚成 1 个类  
filter_mask=np.argwhere(mask)  
kmeans = KMeans(n_clusters=1, random_state=0).fit(filter_mask)  
  
# 获取聚类中心  
cluster_centers_indices = kmeans.cluster_centers_  
# 将聚类中心的索引转换回实际的 (x, y) 坐标  
cluster_centers_x = x[cluster_centers_indices[:, 1].astype(int)]  
cluster_centers_y = y[cluster_centers_indices[:, 0].astype(int)]

# 打印聚类中心的 (x, y) 坐标  
print("聚类中心的 (x, y) 坐标：")  
print(cluster_centers_x, cluster_centers_y)
