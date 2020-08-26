import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import edataset
import numpy as np 
from tqdm import tqdm

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.conv2 = nn.Conv2d(32, 64, 5)
		self.conv3 = nn.Conv2d(64, 128, 5)

		x = torch.randn(50, 50).view(-1, 1, 50, 50)
		self._to_linear = None
		self.convs(x)

		self.fc1 = nn.Linear(self._to_linear, 512)
		self.fc2 = nn.Linear(self._to_linear, 3)
	def convs(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
		x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
		# print(x[0].shape)

		if self._to_linear is None:
			self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
		return x
	def forward(self, x):
		x = self.convs(x)
		x = x.view(-1, self._to_linear)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.softmax(x, dim = 1)



net = Net()
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_function = nn.MSELoss()

h5file = "trial000/data.h5"
word_times_file = "trial000/word-times.csv"
trigger_times_file = "trial000/trigger-times.csv"
training_data = edataset.dataset(h5file, word_times_file, trigger_times_file)
np.random.shuffle(training_data)

X = torch.Tensor([i[1] for i in training_data]).view(-1, 50, 50)
y = torch.Tensor([i[2] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X) * VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

#print(len(train_X))
#print(len(test_X))

BATCH_SIZE = 10
EPOCHS = 10

for epoch in range(EPOCHS):
	for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
		# print(i, i + BATCH_SIZE)
		batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
		batch_y = train_y[i:i+BATCH_SIZE]
		
		net.zero_grad()
		outputs = net(batch_X)
		
		loss = loss_function(outputs, batch_y)
		loss.backward()
		optimizer.step()
		print("LOSS:", loss)
print("OVERALL LOSS", loss)

correct = 0 
total = 0
with torch.no_grad():
	for i in tqdm(range(len(test_X))):
		real_class = torch.argmax(test_y[i])
		net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
		predicted_class = torch.argmax(net_out)
		if predicted_class == real_class:
			correct += 1
		total += 1
print("Accuracy:", round(correct/total, 3))



