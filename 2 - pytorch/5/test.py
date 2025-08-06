import torch
import torchvision
from torch.utils.data import DataLoader



device = torch.device("cuda")

test_data = torchvision.datasets.CIFAR10("./data", False, torchvision.transforms.ToTensor(), download=True)
print(f"测试集长度：{len(test_data)}")

test_dataloader = DataLoader(test_data, 64)

model = torch.load("./model/tudui_25.pth", weights_only=False)
model = model.to(device)

total_correct_cnt = 0

model.eval()
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = model(imgs)

        correct_cnt = (outputs.argmax(1) == targets).sum()
        total_correct_cnt += correct_cnt

print(f"测试集上的正确率：{total_correct_cnt / len(test_data)}")


