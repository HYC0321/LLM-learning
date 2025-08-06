import time
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from model import Tudui

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda")

train_data = torchvision.datasets.CIFAR10("./data", True, torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10("./data", False, torchvision.transforms.ToTensor(), download=True)

print(f"训练集长度：{len(train_data)}")
print(f"测试集长度：{len(test_data)}")

train_dataloader = DataLoader(train_data,64)
test_dataloader = DataLoader(test_data,64)


tudui = Tudui()
tudui = tudui.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(),lr=learning_rate)

total_train_step = 0
total_test_step = 0

epoch = 30

writer = SummaryWriter("./log")
start_time = time.time()
for i in range(epoch):
    print(f"第{i+1}轮训练")

    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"训练次数：{total_train_step}, loss：{loss.item()}, 时间：{end_time - start_time}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_correct_cnt = 0
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data

            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            correct_cnt = (outputs.argmax(1) == targets).sum()
            total_correct_cnt += correct_cnt

    writer.add_scalar("test_loss", total_test_loss, total_train_step)
    writer.add_scalar("test_accuracy", total_correct_cnt / len(test_data), total_train_step)
    print(f"测试集上的Loss：{total_test_loss}")
    print(f"测试集上的正确率：{total_correct_cnt / len(test_data)}")
    torch.save(tudui, f"./model/tudui_{i}.pth")

writer.close()
