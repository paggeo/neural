import torch.nn as nn

class BobNetLinear3(nn.Module):
  def __init__(self):
    super(BobNetLinear3,self).__init__()
    # W x H = 28 x 28
    # Cout = 128
    self.fc1 = nn.Linear(28*28,128)
    # Cout = 128
    self.relu1 = nn.ReLU()
    # Cout = 64
    self.fc2 = nn.Linear(128,64)
    # Cout = 64
    self.relu2 = nn.ReLU()
    # Cout = 10
    self.fc3 = nn.Linear(64,10)
  
  def forward(self, x):
    x = self.relu1(self.fc1(x))
    x = self.relu2(self.fc2(x))
    x = self.fc3(x)
    return x