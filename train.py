import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models import ResNet18_Weights

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

import numpy as np
import pandas as pd
import copy
from datetime import datetime

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def F_make_dir(Path):
    if not os.path.exists(Path):
        os.makedirs(Path)
        

# Training
def train(model, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = train_loss/total
    epoch_acc = correct/total*100
    print("Train | Loss:%.4f Acc: %.2f%% (%s/%s)" 
        % (epoch_loss, epoch_acc, correct, total))
    return epoch_loss, epoch_acc

def test(model, criterion, optimizer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()*inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = test_loss/total
        epoch_acc = correct/total*100
        print("Test | Loss:%.4f Acc: %.2f%% (%s/%s)" 
            % (epoch_loss, epoch_acc, correct, total))
    return epoch_loss, epoch_acc


def add_model_performance(model_name, accuracy, file_path):
    new_data = pd.DataFrame({'Model Name': [model_name], 'Accuracy': [accuracy]})
    try:
        # CSV 파일이 존재하는 경우 기존 데이터에 추가
        existing_data = pd.read_csv(file_path)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        # CSV 파일이 존재하지 않는 경우 새로운 데이터만을 저장
        combined_data = new_data
    combined_data.to_csv(file_path, index=False)

    


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load and preprocess dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    data_dir  = r"./dataset"
    train_dir  =r"%s/train"%(data_dir)
    print(train_dir)
    test_dir  = r"%s/test"%(data_dir)
    model_dir = r"./models"
    model_eval_file = r"./evaluator.csv"
    F_make_dir(train_dir)     
    F_make_dir(test_dir) 
    F_make_dir(model_dir) 
    
    batch_size = 500
        
    trainset = datasets.CIFAR10(root=train_dir,train=True,
                                            download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root=test_dir,train=False,
                                        download=True,transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Load pretrained ResNet model
    #model = torchvision.models.resnet18(pretrained=True)
    model = torchvision.models.resnet18(weights = ResNet18_Weights)

    # Replace the last classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(classes))

    # Freeze all parameters except the last layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    resnet_pt = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    best_acc = 0
    epoch_length = 5
    save_loss = {"train":[],
                "test":[]}
    save_acc = {"train":[],
                "test":[]}

    for epoch in range(epoch_length):
        print("Epoch %s" % epoch)
        train_loss, train_acc = train(resnet_pt, criterion, optimizer)
        save_loss['train'].append(train_loss)
        save_acc['train'].append(train_acc)
        
        if epoch%2 ==0:
            test_loss, test_acc = test(resnet_pt, criterion, optimizer)
            save_loss['test'].append(test_loss)
            save_acc['test'].append(test_acc)

        scheduler.step()

        # Save model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(resnet_pt.state_dict())
        resnet_pt.load_state_dict(best_model_wts)
        
        current_time = datetime.now().strftime("%Y%m%d%H%M")
    torch.save(best_model_wts, '%s/best_model_weights_%s.pth'%(model_dir, current_time))

    add_model_performance('%s/best_model_weights_%s.pth'%(model_dir, current_time), best_acc, model_eval_file)
