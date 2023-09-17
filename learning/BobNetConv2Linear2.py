import torch.nn as nn

class BobNetConv2Linear2(nn.Module):
  def __init__(self):
    super(BobNetConv2Linear2,self).__init__()
    # W x H  = 28 x 28
    # Cout x Wout x Hout = Cin x  [(W_in - K_w + 2 * P) / S] + 1 x [(H_in - K_h + 2 * P) / S] + 1 =  32 x 25 x 25
    self.conv1 = nn.Conv2d(1, 32, kernel_size=4) 
    # Cout x Wout x Hout = 32 x 25 x 25
    self.relu1 = nn.ReLU()
    # Cout x Wout x Hout = Cin x Win/kw x Hin/kh = 32 x 8 (25/3) x 8 (25/3)
    self.pool1 = nn.MaxPool2d(kernel_size=3) 
    # Cout x Wout x Hout = Cin x  [(W_in - K_w + 2 * P) / S] + 1 x [(H_in - K_h + 2 * P) / S] + 1 = 64 x 6 x 6
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
    # Cout x Wout x Hout = 64 x 6 x 6
    self.relu2 = nn.ReLU()
    # Cout = 128 
    self.fc1 = nn.Linear(64 * 6 * 6, 128)
    # Cout = 128 
    self.relu3 = nn.ReLU()
    # Cout = 10 
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    x = self.relu1(self.conv1(x))
    x = self.pool1(x)
    x = self.relu2(self.conv2(x))
    x = self.relu3(self.fc1(x.view(x.size(0),-1)))
    x = self.fc2(x)
    return x