import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hàm Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Load data từ file csv bằng pandas
data = pd.read_csv('dataset.csv').values
N, d = data.shape
x = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)

# Vẽ đồ thị scatter bằng matplotlib
plt.scatter(x[:10, 0], x[:10, 1], c='red', edgecolors='none', s=30, label='cho vay')
plt.scatter(x[10:, 0], x[10:, 1], c='blue', edgecolors='none', s=30, label='Khong cho')
plt.legend(loc=1)
plt.xlabel('Mức lương')
plt.ylabel('Kinh nghiệm')
#plt.show()

# Thêm cột vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x))

w = np.array([0.,0.1,0.1]).reshape(-1, 1)

# Số lần lặp bước 2 của thuật toán Gradient descent
num = 1000
cost = np.zeros((num,1))
learning_rate = 0.01

for i in range(1, num):
    # Tính giá trị dự đoán bằng công thức logistic
    y_predict = sigmoid(np.dot(x, w))
    # Loss function
    cost[i] = -np.sum(np.multiply(y, np.log(y_predict)) + np.multiply(1-y, np.log(1-y_predict)))
    # Gradient descent
    w -= learning_rate*np.dot(x.T, y_predict - y)

# T là tỉ lệ cho vay của công ti
t = 0.5
plt.plot((4, 10), (-(w[0] + 4*w[1] + np.log(1/t-1)) / w[2], -(w[0] + 10*w[1] + np.log(1/t-1)) / w[2]), 'g')
plt.show()

x1 = 4
x2 = 1
y1 = w[0] + x1*w[1] + x2*w[2]
print(y1)
if(y1> -np.log(1/t-1)):
    print('Cho vay')
else:
    print('Khong cho vay')