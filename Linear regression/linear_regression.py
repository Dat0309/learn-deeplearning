import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# numPoint = 30
# noise = np.random.normal(0,1,numPoint).reshape(-1,1)
# x = np.linspace(30, 100, numPoint).reshape(-1,1)
# N = x.shape[0]
# y = 15*x + 8 + 20*noise
# plt.scatter(x, y)
# plt.show()

'''
Bài toán dự đoán giá nhà dùng tập dữ liệu data_linear.csv
'''
# Đọc dữ liệu trong data_linear.csv bằng pandas
data = pd.read_csv('data_linear.csv').values
# Lấy dữ liệu đầu
N = data.shape[0]
# print(N)
# Tách mảng thành nhiều mảng, mỗi mảng chứa 1 điểm x hoặc y, dùng để biểu diễn tọa độ 
# trên sơ đồ trực quan hóa dữ liệu: diện tích: x, giá tiền: y
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

# Biểu diễn x, y bằng biểu đồ phân tán scatter
plt.scatter(x, y)
plt.xlabel('met vuong')
plt.ylabel('gia')

# Nối chuỗi các mảng theo chiều ngang
x = np.hstack((np.ones((N, 1)), x))

w = np.array([0.,1.]).reshape(-1, 1)

# print('x----------\n', x)
# print('y----------\n', y)
# print('w----------\n', w)

numOfIteration = 100
cost = np.zeros((numOfIteration, 1))
learing_rate = 0.000001

# r = np.dot(x, w) - y
# print('r----------\n',r)
# print(np.sum(r))
# print(np.sum(np.multiply(r,x[:,1].reshape(-1,1))))

# print('x:1----------------',x[:,1].reshape(-1,1))
# print('multiply',np.multiply(r,x[:,1].reshape(-1,1)))
# cost[1] = 0.5*np.sum(r*r)
# w[0] -= learing_rate*np.sum(r)
# w[1] -= learing_rate*np.sum(np.multiply(r,x[:,1].reshape(-1,1)))
# print('w0-------',w[0])
# print('w1-------',w[1])

# print('cost------------',cost[1])

'''
Áp dụng thuật toán Gradient descent tìm giá trị nhỏ nhất của hàm số f(x) dựa trên đạo hàm
x = x - learning_rate*f'(x)
Bước 1: Khởi tạo x = x0 tùy ý
Bước 2: Gán x = x - learning_rate*f'(x)
Bước 3: Tính lại f(x): Nếu f(x) đủ nhỏ thì dừng lại, ngược lại tiếp tục bước 2
'''
for i in range(1, numOfIteration):
    # Tích vô hướng 2 mảng
    r = np.dot(x, w) - y
    cost[i] = 0.5*np.sum(r*r)
    w[0] -= learing_rate*np.sum(r)
    w[1] -= learing_rate*np.sum(np.multiply(r,x[:,1].reshape(-1,1)))
    print(cost[i])
predict = np.dot(x, w)
#print('predict\n',predict)
plt.plot((x[0][1], x[N-1][1]), (predict[0], predict[N-1]), 'r')
plt.show()
plt.savefig('linear_regression.png')

x1 = 50
y1 = w[0] + w[1]*x1
print('Gia nha 50m vuong la: ', y1)