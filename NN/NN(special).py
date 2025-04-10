import torch
import math as m
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 定义神经网络结构
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 生成数据
def generate_data(num_samples = 100):
    #torch.linspace(start, end, steps)函数，用于生成一个一维张量，
    # 其中包含从 start 到 end（包括这两个值）的 steps 个均匀间隔的点。
    #.unsqueeze(1)：这是一个 PyTorch 张量操作，用于增加张量的维度。
    # unsqueeze(1) 表示在张量的第1个维度（从0开始计数）增加一个新的维度。
    x = torch.linspace(-10, 10, num_samples).unsqueeze(1)
    y = 6 * x**3 + 8 * x**2 + 9 * x + 0.1
    return x, y

# 实例化网络
input_size = 1
hidden_size = 256
num_classes = 1
net = SimpleNet(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)

# 训练网络
num_epochs = 1000
x, y = generate_data()

for epoch in range(num_epochs):
    # 前向传播
    outputs = net(x)
    loss = criterion(outputs, y)
    
    # 反向传播和优化
    #梯度清零，防止累加。
    optimizer.zero_grad()
    #反向传播计算梯度。
    loss.backward()
    #这一行代码根据计算出的梯度更新模型的参数。
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试网络
with torch.no_grad():
    predicted = net(x).data.numpy()
    actual = y.data.numpy()

# 绘制结果
plt.scatter(x.data.numpy(), actual, label='Actual')
plt.plot(x.data.numpy(), predicted, color='red', label='Predicted')
plt.legend()
plt.show()