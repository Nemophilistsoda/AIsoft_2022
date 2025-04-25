import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import KBinsDiscretizer  # 用于连续特征离散化

file_path = 'F:\\Competition_2025\\BDAIAC\\AIsoft_2022\\data\\task3\\task3data.csv'
data = np.loadtxt(file_path, delimiter=',', encoding='utf-8')
print(data.shape)

print('标签列示例值', data[:5, 30])
discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
discrete_labels = discretizer.fit_transform(data[:, 30].reshape(-1, 1)).ravel()

# 取对前30维特征进行归一化
data_std = np.zeros_like(data[:, :30])  # 创建一个与前30维特征相同形状的全零数组
for i in range(30):
    col_min = data[:, i].min()
    col_max = data[:, i].max()
    data_std[:, i] = (data[:, i] - col_min) / (col_max - col_min)

# 切分数据集：前500行训练集，后69行测试集
train_data = np.column_stack((data_std[:500, :], data[:500, 30]))  # 前500行: 30列归一化特征 + 原始标签列
test_data = np.column_stack((data_std[500:, :], data[500:, 30]))

# 使用`sklearn`的SVM模型训练（核函数为多项式、维度4、惩罚系数0.5）
clf = SVC(kernel='poly', degree=4, C=0.5)
'''
kernel: 核函数类型，常用的有 'linear'（线性核）、'poly'（多项式核）、'rbf'（径向基函数核）等。
degree: 多项式核的次数，当核函数为多项式时有效。
C: 惩罚系数，用于控制模型的复杂度。C值越大，模型对训练数据的拟合程度越高，但可能导致过拟合。
'''
clf.fit(train_data[:, :30], train_data[:, 30])
# def fit(self, X, y, sample_weight=None):
# 所以train_data[:, :30]是训练数据的特征部分，即X
# train_data[:, 30]是训练数据的标签部分，即y
'''
train_data[:, :30]：训练数据的特征部分，即前30列。
train_data[:, 30]：训练数据的标签部分，即第31列。
clf.fit() 方法用于训练模型，将特征和标签作为输入，并根据指定的核函数、维度和惩罚系数进行模型的训练。
训练完成后，模型会根据训练数据的特征和标签进行学习，从而得到一个可以用于预测的模型。
'''
# 预测测试集的标签
predictions = clf.predict(test_data[:, :30])
# 输出预测结果
for i, prediction in enumerate(predictions):
    print(f"第{i+1}个测试样本的预测标签为：{prediction}")

# 计算准确率
accuracy = np.mean(predictions == test_data[:, 30])
print(f"准确率为：{accuracy}")