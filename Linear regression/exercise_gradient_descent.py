import numpy as np
'''
Áp dụng Gradient descent tìm giá tị nhỏ nhất của hàm số X**2 + 2X + 5
'''
x = 6
learning_rate = 0.0001
precision = 0.000001
previous_step_size = x

def df(x):
    y = 2*x + 2
    return y

while previous_step_size > precision:
    prev_x = x
    x += - learning_rate*df(prev_x)
    previous_step_size = abs(x - prev_x)
    print(x)

print('minimum y = ', x)
