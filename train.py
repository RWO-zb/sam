import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
import numpy as np
import os
from sam import SAM
from resnet18_sam import ResNet18_SAM

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders():
    """获取训练和测试数据加载器"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_epoch(model, optimizer, criterion, trainloader, use_sam=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        def closure():
            nonlocal correct, total
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            return loss

        if use_sam:
            loss = closure()
            optimizer.first_step(zero_grad=True)
            closure()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            loss = closure()
            optimizer.step()
        
        total_loss += loss.item()
    
    train_acc = 100. * correct / total
    return total_loss / len(trainloader), train_acc

def test_model(model, testloader):
    """测试模型性能"""
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total, total_loss / len(testloader)

def run_experiment1(opt_name="SGD", num_epochs=10):
    """运行完整实验"""
    trainloader, testloader = get_dataloaders()
    model = ResNet18_SAM(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4, amsgrad=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif opt_name == "SAM":
        base_optimizer = optim.SGD
        params = list(model.parameters())
        optimizer = SAM(params, base_optimizer, rho=0.1, lr=0.05, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer.base_optimizer, step_size=30, gamma=0.1)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, criterion, trainloader, use_sam=(opt_name == "SAM"))
        test_acc, test_loss = test_model(model, testloader)
        
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        
        print(f"[{opt_name}] Epoch {epoch+1:2d}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        scheduler.step()
    
    # 保存训练数据
    data = np.column_stack((
        metrics['train_loss'], 
        metrics['train_acc'], 
        metrics['test_loss'], 
        metrics['test_acc']
    ))
    np.savetxt(f'{opt_name}_metrics.txt', data, delimiter=',', 
               header='train_loss,train_acc,test_loss,test_acc', comments='')
    
    return metrics

# train.py (修改部分)
def run_experiment2(opt_name="SGD", num_epochs=10, params_list=None):
    """运行完整实验，支持多参数配置"""
    if params_list is None:
        if opt_name == "SGD":
            params_list = [{'lr': 0.1, 'momentum': 0.9}]
        elif opt_name == "Adam":
            params_list = [{'lr': 3e-4}]
        elif opt_name == "SAM":
            params_list = [{'rho': 0.1, 'lr': 0.05, 'momentum': 0.9}]
    
    trainloader, testloader = get_dataloaders()
    all_metrics = {}
    
    for i, params in enumerate(params_list):
        model = ResNet18_SAM(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()

        # 关键修复：将参数转换为列表并验证
        model_params = list(model.parameters())
        if not model_params:
            raise ValueError(f"Model parameters are empty for {opt_name} Config {i+1}")
        
        print(f"\n[Training {opt_name} Config {i+1}]")
        print(f"Parameters on: {model_params[0].device}")
        print(f"Total parameters: {sum(p.numel() for p in model_params)}")

        metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        # 优化器配置
        if opt_name == "SGD":
            optimizer = optim.SGD(model_params,
                               lr=params['lr'],
                               momentum=params.get('momentum', 0.9),
                               weight_decay=5e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif opt_name == "Adam":
            optimizer = optim.Adam(model_params,
                                lr=params['lr'],
                                weight_decay=5e-4,
                                amsgrad=True)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        elif opt_name == "SAM":
            optimizer = SAM(model_params,
                          base_optimizer=optim.SGD,
                          rho=params.get('rho', 0.1),
                          lr=params.get('lr', 0.05),
                          momentum=params.get('momentum', 0.9),
                          weight_decay=5e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer.base_optimizer, step_size=30, gamma=0.1)

        # 训练循环
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, optimizer, criterion, trainloader, use_sam=(opt_name == "SAM"))
            test_acc, test_loss = test_model(model, testloader)
            
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Acc: {train_acc:6.2f}% | "
                  f"Test Loss: {test_loss:.4f} | Acc: {test_acc:6.2f}%")
            
            scheduler.step()
        
        # 保存结果
        config_id = f"{opt_name}_config_{i+1}"
        all_metrics[config_id] = metrics
        np.savetxt(f'{config_id}_metrics.txt', 
                  np.column_stack((metrics['train_loss'], metrics['train_acc'], 
                                 metrics['test_loss'], metrics['test_acc'])),
                  delimiter=',', header='train_loss,train_acc,test_loss,test_acc')
    
    return all_metrics