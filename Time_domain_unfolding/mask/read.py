# # import scipy.io
# # import matplotlib.pyplot as plt
# #
# # data = scipy.io.loadmat('mask.mat')  # 加载文件
# # # 查看变量名：print(data.keys())，找到目标变量（排除'__header__'等系统变量）
# # hadamard_mats = data['mask']  # 假设是三维数组
# #
# # for i in range(20):
# #     plt.figure()
# #     plt.imshow(hadamard_mats[:, :, i], cmap='gray')
# #     plt.title(f'哈达玛矩阵图 {i+1}')
# #     plt.axis('off')
# #     plt.savefig(f'hadamard_{i+1}.png', bbox_inches='tight')
# # plt.show()
#
#
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def hadamard(n):
#     """生成n×n的哈达玛矩阵"""
#     # 初始化为2×2矩阵
#     H = np.array([[1, 1], [1, -1]])
#
#     # 通过递归构造更大的矩阵
#     while H.shape[0] < n:
#         H = np.vstack([np.hstack([H, H]), np.hstack([H, -H])])
#
#     return H[:n, :n]
#
#
# # 生成8×8的哈达玛矩阵
# n = 8
# H = hadamard(n)
#
# # 将-1转换为0以便于可视化
# H_visual = (H + 1) // 2
#
# # 显示哈达玛矩阵
# plt.figure(figsize=(6, 6))
# plt.imshow(H_visual, cmap='binary')
# plt.axis('off')
# plt.title(f'{n}×{n} Hadamard Matrix')
# plt.show()

import numpy as np

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 定义矩阵大小
size = 200

# 生成随机矩阵（0为黑色，1为白色）
matrix = np.random.randint(0, 2, (size, size))
