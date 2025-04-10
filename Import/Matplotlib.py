import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False      #解决中文乱码问题

"""
一般函数图像
"""
X = np.linspace(-10, 10, 200)
Y = 2 * X
U = np.sin(X)
plt.plot(X, Y, label = "$y = 2*x$")
plt.plot(X, U, label = "$u = sin(x)$")
plt.title('函数图像', fontsize = 15)
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.legend(fontsize = 14)
plt.grid(True)
plt.show()

"""
散点图(一维)
"""
X = np.random.rand(50)
Y = np.random.rand(50)
plt.scatter(X, Y)
plt.title('散点图', fontsize = 15)
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.legend()
plt.show()

"""
特征数据点
"""
# 创建示例数据
np.random.seed(0)
n_samples = 50
x1_class0 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)  # 特征 x1
x2_class0 = np.random.normal(loc=0.0, scale=1.0, size=n_samples)  # 特征 x2
labels_class0 = np.zeros(n_samples)  # 标签为 0
x1_class1 = np.random.normal(loc=3.0, scale=1.0, size=n_samples)  # 特征 x1
x2_class1 = np.random.normal(loc=3.0, scale=1.0, size=n_samples)  # 特征 x2
labels_class1 = np.ones(n_samples)  # 标签为 1
x1 = np.concatenate([x1_class0, x1_class1])
x2 = np.concatenate([x2_class0, x2_class1])
labels = np.concatenate([labels_class0, labels_class1])

plt.scatter(x1[labels == 0], x2[labels == 0], label = "类型一")
plt.scatter(x1[labels == 1], x2[labels == 1], label = "类型一")

plt.legend()
plt.grid()
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

# 创建示例数据
data = {
    'X': np.random.rand(50),
    'Y': np.random.rand(50),
    'Z': np.random.rand(50)
}
df = pd.DataFrame(data)

# 创建图形和 3D 轴
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(df['X'], df['Y'], df['Z'], c='blue', marker='o')

# 添加拟合平面（可选）
# 假设我们已经拟合了一个平面方程 Z = aX + bY + c
a, b, c = 1, 1, 0  # 示例参数
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)
Z = a * X + b * Y + c

# 绘制平面
ax.plot_surface(X, Y, Z, color='red', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Scatter Plot with Fitted Plane')
plt.show()