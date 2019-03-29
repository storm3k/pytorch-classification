import torch as t
from torch import nn
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet34
from torchvision import transforms as T

from data import DogCat
from config import opt


class MyTrain(object):

    def __init__(self):
        pass

    def get_dataset(self):
        print('==> Preparing data..')

        transform_train = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = ImageFolder(opt.train_data_root, transform=transform_train)
        test_set = ImageFolder(opt.test_data_root, transform=transform_test)
        print("train_set: ", train_set.class_to_idx)
        print("train_set: %d imgs" % len(train_set.imgs))
        print("test_set: ", test_set.class_to_idx)
        print("test_set: %d imgs" % len(test_set.imgs))

        train_loader = DataLoader(
            train_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers
        )
        test_loader = DataLoader(
            test_set,
            batch_size=opt.test_batch_size,
            shuffle=False,
            num_workers=opt.num_workers
        )

        return train_loader, test_loader

    def get_model(self, device):
        print('==> Building model..')
        if device == 'cuda':
            cudnn.benchmark = True
        return resnet34()

    def get_optimizer(self, model):
        # 目标函数
        criterion = nn.CrossEntropyLoss()
        # 优化方法
        optimizer = optim.SGD(model.parameters(),
                              opt.lr,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)
        return criterion, optimizer

    def meters(self):
        pass

    def train(self, net, epoch, device, data_loader, optimizer, criterion):
        print('\nEpoch: %d' % (epoch+1))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        print('train acc %.3f' % accuracy)

        return accuracy

    def test(self, net, device, data_loader, criterion):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with t.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        print(' test acc %.3f' % accuracy)

        return accuracy

    def run(self):
        # step1: data
        train_loader, test_loader = self.get_dataset()
        device = 'cuda' if t.cuda.is_available() else 'cpu'

        # step2: configure model
        net = self.get_model(device)
        net.to(device)

        # step3: criterion and optimizer
        criterion, optimizer = self.get_optimizer(net)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[60, 80],
                                                   gamma=0.1)

        # 记录 accuracy
        train_accuracies = []
        test_accuracies = []

        for epoch in range(opt.max_epoch):
            scheduler.step()
            train_acc = self.train(net, epoch, device, train_loader, optimizer, criterion)
            test_acc = self.test(net, device, test_loader, criterion)


if __name__ == '__main__':
    my = MyTrain()
    my.run()
