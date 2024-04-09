import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import linregress

# 参数设定
A =  1 # 假设的参数A, normalized extrusion prob Omega
B = -0.02 # 假设的参数B, normalized Peclet number -(s0-0.5)*Pe/2d
Bsi = 1.0
epsilon = 1e-5  # 增加 epsilon 的值

# 网格的设定
x_min, x_max = 0.001, 100
y_min, y_max = 0.001, 100
num_points = 10000
x = np.logspace(np.log10(x_min), np.log10(x_max), num_points)
y = np.logspace(np.log10(y_min), np.log10(y_max), num_points)

# C(x)的初始猜测设为 x^{-1/2}
C_x = A * 1 / np.sqrt(np.abs(x) + epsilon)  # 添加 epsilon 防止除以零

# 迭代求解
max_iter = 50  # 迭代次数
for _ in range(max_iter):
    new_C_x = np.zeros_like(C_x)
    R_x_values = np.zeros_like(x)
    for i, xi in enumerate(x):
        # 定义积分函数
        def integrand(y):
            return C_x[i] * (xi - y)

        # 计算积分
        integral_part, _ = quad(integrand, 0, xi, limit=100, epsabs=1e-4, epsrel=1e-4)

        # 定义积分函数
        def integrand(y):
            return C_x[i]

        # 计算积分
        integral_part1, _ = quad(integrand, 0, 1, limit=100, epsabs=1e-4, epsrel=1e-4)

        # 计算R(x)
        R_x_values[i] = abs(xi) * (1-2*B) + B * integral_part

        # 更新C(x)
        new_C_x[i] = A * (R_x_values[i] ** (-3 / 2))

    C_x = new_C_x


C_x=C_x/A

# 绘制C(x)随x的变化曲线
plt.loglog(x, C_x)
plt.xlabel('x')
plt.ylabel('C(x)')
plt.title('C(x) vs x')
#plt.grid(True)
plt.show()

# 修改 X 轴和 Y 轴标注的字体
fontdict = {'size': 20, 'weight': 'bold'}

#plt.figure(figsize=(12, 8))
# 绘制C_x的log-log图
#plt.subplot(1, 1, 1)
#plt.loglog(x, C_x, color='orange', marker='o')
#plt.loglog(x, C_x, color='limegreen', marker='o')
plt.loglog(x, C_x, color='tomato', marker='o')
#plt.loglog(x, C_x, color='deepskyblue', marker='o')
plt.xlabel('', fontdict=fontdict)
plt.ylabel('', fontdict=fontdict)
# 调整x轴和y轴坐标值的大小
plt.tick_params(axis='both', which='major', labelsize=20)  # 修改主刻度标签的大小

plt.savefig('/Chromosome/Fig/uni3.svg')
#plt.savefig('/Chromosome/Fig/uni4.png')
# 组合 x 和 C_x 数据
data = np.column_stack((x, C_x))

# 指定保存路径
file_path = '/Chromosome/x_Cx_data_uni.txt'

# 将数据保存为txt文件
np.savetxt(file_path, data, fmt='%10.5f', header='x C(x)')

file_path


# 拟合C_x在x=0附近的斜率
# 选取非零值进行拟合
non_zero_indices = C_x > 0
log_x = np.log(x[non_zero_indices])
log_C_x = np.log(C_x[non_zero_indices])
slope, intercept, r_value, p_value, std_err = linregress(log_x, log_C_x)

print("斜率拟合结果：", slope)

# 计算仅包括前两个点的斜率
# 确保前两个点的C_x值不为零
if C_x[0] > 0 and C_x[1] > 0:
    log_x_1, log_x_2 = np.log(x[0]), np.log(x[1])
    log_C_x_1, log_C_x_2 = np.log(C_x[0]), np.log(C_x[1])
    slope_2_points = (log_C_x_2 - log_C_x_1) / (log_x_2 - log_x_1)
    print("仅前两个点的斜率拟合结果：", slope_2_points)
else:
    print("前两个点的C_x值不适合进行拟合")

plt.show()

