import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_square.csv').values
N = data.shape[0]

x = data[:,0].reshape(-1, 1)
y = data[:,1].reshape(-1, 1)

plt.scatter(x,y)
plt.xlabel('dien tich')
plt.ylabel('gia')
plt.show()