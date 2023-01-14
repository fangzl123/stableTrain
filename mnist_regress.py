#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root = './mnist_data'
if not os.path.exists(root):
    os.mkdir(root)
    
    
trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), 
                            transforms.Normalize((0.1307,), (0.3081,))])
# load mnist dataset: train_set = 60000, test_set = 10000, image size = 28*28
train_set = torchvision.datasets.MNIST(root=root, train=True, transform=trans, download=True)
test_set = torchvision.datasets.MNIST(root=root, train=False, transform=trans, download=True)

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)
    

# print('==>>> total trainning batch number: {}'.format(len(train_loader)))
# print('==>>> total testing batch number: {}'.format(len(test_loader)))

## network

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)    
        self.conv2 = nn.Conv2d(6, 16, 5, 1)   
        self.conv3 = nn.Conv2d(16, 120, 5, 1) 
        self.fc1 = nn.Linear(1*1*120, 84)
        self.fc2 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(10, 1)  # output single prediction value y_hat


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1*1*120)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        self.feature_out = x
        x = self.fc3(x)
        return x
    
    def name(self):
        return "LeNet5"
    
# calculate gradients manually

def calculate_grad_L2(feature_data, out, target):
    delta_y = out - target
    extend_y_matrix = delta_y.reshape([-1,1]).expand(delta_y.size(0), feature_data.size(1))
    gradw_matrix = torch.mul(feature_data, extend_y_matrix)
    
    grad_w = 2 * torch.mean(gradw_matrix, 0)
    grad_b = 2 * torch.mean(delta_y)
    
    return grad_w.reshape([-1, feature_data.size(1)]), grad_b.reshape([-1])

    
def calculate_grad_L1(feature_data, out, target):
    pred_sign = torch.sign(out - target)
    extend_y_matrix = pred_sign.reshape([-1,1]).expand(pred_sign.size(0), feature_data.size(1))
    gradw_matrix = torch.mul(feature_data, extend_y_matrix)
    
    grad_w = torch.mean(gradw_matrix, 0)
    grad_b = torch.mean(pred_sign)
    
    return grad_w.reshape([-1, feature_data.size(1)]), grad_b.reshape([-1])
    
# test & train function

def exe_test(model, test_loader, criterion, regular_term, lambda_reg, accuracy_res):
    correct_cnt, test_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        target = target.type(torch.float32)
        
        with torch.no_grad():
            out = model(x)
            out = out.reshape([-1])
            loss = criterion(out, target) + lambda_reg * regular_term 
            # _, pred_label = torch.max(out.data, 1)    # max probability when classification
            pred_label = out.data
            total_cnt += x.data.size()[0]
            # correct_cnt += (pred_label == target.data).sum()
            correct_cnt += (torch.abs(pred_label - target.data) <= 0.45).sum()

        test_loss += loss.item()
        
        # if (batch_idx+1) == len(test_loader):
        #     print('==>>> test loss: {:.6f}, accuracy: {:.4f}'.format(
        #         test_loss/len(test_loader), correct_cnt * 1.0 / total_cnt))
    
    accuracy_res.append(correct_cnt.item() * 1.0 / total_cnt)
    

def construct_A_matrix(model_layer, current_lr):
    # num_in = model_layer.in_features
    num_out = model_layer.out_features
    
    grad_fc = torch.cat([model_layer.weight.grad, model_layer.bias.grad.view(num_out,1)], 1)
    change_term = torch.transpose(-grad_fc * current_lr, 0, 1)
    
    A_row1 = torch.cat([torch.eye(grad_fc.size(1)).to(device),change_term], 1)
    A_row2 = torch.cat([torch.zeros(num_out,A_row1.size(0)).to(device), torch.eye(num_out).to(device)], 1)
    
    A_matrix = torch.cat([A_row1, A_row2], 0)
    
    return A_matrix

def construct_A_matrix_manually(model_layer, grad_weight, grad_bias, current_lr):
    # num_in = model_layer.in_features
    num_out = model_layer.out_features
    
    grad_fc = torch.cat([grad_weight, grad_bias.view(num_out,1)], 1)
    change_term = torch.transpose(-grad_fc * current_lr, 0, 1)
    
    A_row1 = torch.cat([torch.eye(grad_fc.size(1)).to(device),change_term], 1)
    A_row2 = torch.cat([torch.zeros(num_out,A_row1.size(0)).to(device), torch.eye(num_out).to(device)], 1)
    
    A_matrix = torch.cat([A_row1, A_row2], 0)
    
    return A_matrix

    
def exe_train(model, train_loader, criterion, learning_rate, lambda_reg, accuracy_res):
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    
    true_C_matrix_fc3 = torch.eye(model.fc3.in_features + model.fc3.out_features + 1).to(device)
    
    compare_term = torch.zeros([1], dtype=torch.float32, device=device, requires_grad=True)
    regular_term = torch.zeros([1], dtype=torch.float32, device=device, requires_grad=True)
    
    grad_diff_res = []
    
    epoch_num = 10
    for epoch in range(epoch_num):    
        # trainning
        train_loss = 0
        for batch_idx, (x, target) in enumerate(train_loader):
            x, target = x.to(device), target.to(device)
            target = target.type(torch.float32)
            
            optimizer.zero_grad()
            out = model(x) 
            out = out.reshape([-1])
            
            feature_fc2 = model.feature_out
            
            loss = criterion(out, target)  
            loss.backward(retain_graph=True)  # also: torch.autograd.backward(loss)
            # default: retain_/create_graph=False: the graph used to compute the grads will be freed
              
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            # print(current_lr)
            
            
            # construct A matrix
            # A_matrix_fc3 = construct_A_matrix(model.fc3, current_lr)
            grad_weight, grad_bias = calculate_grad_L1(feature_fc2, out, target)
            A_matrix_fc3 = construct_A_matrix_manually(model.fc3, grad_weight, grad_bias, current_lr)
            
            # # difference between manual & automatic calculation
            mix_grad = torch.cat([grad_weight, grad_bias.view(1,1)],1)
            mix_autograd = torch.cat([model.fc3.weight.grad, model.fc3.bias.grad.view(1,1)],1)
            _diff = mix_grad - mix_autograd
            grad_diff = _diff.clone().detach().cpu().numpy()
            grad_diff_res.append(grad_diff)
            
            # multiply with history & calculate eigenvalues
            C_matrix_fc3 = torch.mm(true_C_matrix_fc3, A_matrix_fc3)
            
            _, sig_fc3, _ = torch.svd(C_matrix_fc3)   # singular values are descending
            joint_spectral_fc3 = sig_fc3[0]
                      
            regular_term_fc3 = torch.max(compare_term, joint_spectral_fc3 - 1)
        
            
            # # backward  again with regular loss, add into original loss
            reg_loss = lambda_reg * regular_term_fc3 
            reg_loss.backward()
        
            optimizer.step()
            
            
            # use true A_matrix (contains grads of regular loss) to construct history
            updated_A_matrix_fc3 = construct_A_matrix(model.fc3, current_lr)          
            true_C_matrix_fc3 = torch.mm(true_C_matrix_fc3, updated_A_matrix_fc3)
            
            
            # final loss function
            final_loss = loss.item() + reg_loss.item()
            train_loss += final_loss
            
            
            if (batch_idx+1) % 600 == 0:
                print('==>>> train loss: {:.6f}'.format(train_loss/len(train_loader)))
    
            if (batch_idx+1) % 20 == 0:
                exe_test(model, test_loader, criterion, regular_term, lambda_reg, accuracy_res)
                
    grad_diff_res = np.array(grad_diff_res).reshape([len(grad_diff_res),11])
    np.savetxt('grad_diff_0130.csv', grad_diff_res,delimiter=',')

#%%    
    
model = LeNet5().to(device)
criterion = nn.L1Loss()

learning_rate = 0.1
lambda_reg = 30

# spectral_res = []
accuracy_res = []

exe_train(model, train_loader, criterion, learning_rate, lambda_reg, accuracy_res)
 

np.savetxt('l1_lr01_reg30.txt',np.array(accuracy_res),fmt='%.6e')
# torch.save(model.state_dict(), model.name())

#%%

set_lr = [0.05, 0.1, 0.15]  # 0.05, 0.09, 0.16, 0.28, 0.5, 0.2, 0.36, 0.63
set_reg = [0, 1, 3, 10, 30, 100] 


for learning_rate in enumerate(set_lr):
    for lambda_reg in enumerate(set_reg):
        for i in range(5):
            model = LeNet5().to(device)
            criterion = nn.L1Loss()
            
            accuracy_res = []
            exe_train(model, train_loader, criterion, learning_rate[1], lambda_reg[1], accuracy_res)
            file_name = 'l1_data_man/lr' + str(learning_rate[1]) +'_' + 'reg' + str(lambda_reg[1]) + '_' + str(i) + '.txt'
            np.savetxt(file_name, np.array(accuracy_res), fmt='%.6e')
            
            del model
            del accuracy_res
            