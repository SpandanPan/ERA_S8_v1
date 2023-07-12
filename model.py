from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class Model_Norm(nn.Module):
    def __init__(self,norm_type):
        super(Model_Norm, self).__init__()
        self.norm_type=norm_type
        # Conv Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm1=nn.BatchNorm2d(16)
        elif norm_type=='ln':
          self.norm1=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm1=nn.GroupNorm(num_groups=2,num_channels=10)
        self.dp1=nn.Dropout(p=0.1)

       # Conv Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm2=nn.BatchNorm2d(16)
        elif norm_type=='ln':
          self.norm2=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm2=nn.GroupNorm(num_groups=4,num_channels=10)
        self.dp2=nn.Dropout(p=0.1)

       # TB Conv Layer1 1*1
        self.conv_tb1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
       # Max Pool Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)

      # Conv Layer 3
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=24, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm3=nn.BatchNorm2d(24)
        elif norm_type=='ln':
          self.norm3=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm3=nn.GroupNorm(num_groups=4,num_channels=10)
        self.dp3=nn.Dropout(p=0.1)

      # Conv Layer 4
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm4=nn.BatchNorm2d(24)
        elif norm_type=='ln':
          self.norm4=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm4=nn.GroupNorm(num_groups=4,num_channels=10)
        self.dp4=nn.Dropout(p=0.1)

       # Conv Layer 5
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm5=nn.BatchNorm2d(24)
        elif norm_type=='ln':
          self.norm5=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm5=nn.GroupNorm(num_groups=4,num_channels=10)
        self.dp5=nn.Dropout(p=0.1)

      # TB Conv Layer2 1*1
        self.conv_tb2 = nn.Conv2d(in_channels=24, out_channels=8, kernel_size=(1, 1), padding=0, bias=False)
       # Max Pool Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)

       # Conv Layer 6
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm6=nn.BatchNorm2d(32)
        elif norm_type=='ln':
          self.norm6=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm6=nn.GroupNorm(num_groups=4,num_channels=10)
        self.dp6=nn.Dropout(p=0.1)

      # Conv Layer 7
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm7=nn.BatchNorm2d(32)
        elif norm_type=='ln':
          self.norm7=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm7=nn.GroupNorm(num_groups=4,num_channels=10)
        self.dp7=nn.Dropout(p=0.1)

       # Conv Layer 8
        self.conv8 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False)
        if norm_type=='bn':
          self.norm8=nn.BatchNorm2d(32)
        elif norm_type=='ln':
          self.norm8=nn.GroupNorm(num_groups=1,num_channels=10)
        elif norm_type=='gn':
          self.norm8=nn.GroupNorm(num_groups=4,num_channels=10)
        self.dp8=nn.Dropout(p=0.1)

       # GAP Layer
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=8))

       # Output Layer
        self.conv9 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)


    def forward(self, x):
        x = self.dp1(self.norm1(F.relu(self.conv1(x))))
        x = x+self.dp2(self.norm2(F.relu(self.conv2(x))))
        x = self.pool1(self.conv_tb1(x))
        x = self.dp3(self.norm3(F.relu(self.conv3(x))))
        x = x+self.dp4(self.norm4(F.relu(self.conv4(x))))
        x = x + self.dp5(self.norm5(F.relu(self.conv5(x))))
        x=  self.pool2(self.conv_tb2(x))
        x = self.dp6(self.norm6(F.relu(self.conv6(x))))
        x = x+self.dp7(self.norm7(F.relu(self.conv7(x))))
        x = x+self.dp8(self.norm8(F.relu(self.conv8(x))))
        x = self.gap(x)
        x = self.conv9(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
    return test_loss

