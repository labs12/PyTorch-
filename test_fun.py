#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_utils import train, test
import matplotlib.pyplot as plt



def train(model,train_loader,loss_func,optimizer,step,device,print_step=200):
    '''Train Function'''
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = loss_func(output,target)
        loss.backward()
        optimizer.step()
        #to print the middle steps
        if batch_idx % print_step == 0:
            print('Train Step: {} ({:05.2f}%)  \tLoss: {:.4f}'.format(
                step, 100.*(batch_idx*train_loader.batch_size)/len(train_loader.dataset), 
                loss.item()))
            
            
def test(model,test_loader,loss_func,device):
    '''test function'''
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device), target.to(device)
            output = model(data)
            #calculate loss 
            test_loss += loss_func(output,target,reduction="sum").item()
            pred = output.softmax(1).argmax(dim=1,keepdim=True)
            #calculate the accurate prediction
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss/= len(test_loader.dataset)
    test_acc = correct/len(test_loader.dataset)
    print('Test Set: Average loss:{:.4f},Accuracy:{}/{}({:05.2f}%)'.format(
    test_loss,correct,len(test_loader.dataset),100.*test_acc))
    return test_loss, test_acc


def main(model,train_loader,test_loader,loss_func,optimizer,n_step,device,save_path,print_step):
    
    test_accs = []
    best_acc = 0.0
    
    for step in range(1,n_step+1):
        #training 
        train(model,train_loader,loss_func, optimizer,
             step=step,device=device,print_step=print_step)
        #evaluation
        test_loss, test_acc = test(model,test_loader,loss_func=F.cross_entropy,device=device)
        
        #to keep record of accuracy 
        test_accs.append(test_acc)
        #to decide wether to save the optimal parameter/test results or not 
        if len(test_accs)>=2:
            if test_acc >= best_acc:
                best_acc = test_acc
                best_state_dict = model.state_dict()
                print("discard previous state, best model state saved!")
        print("")
        
    torch.save(best_state_dict,save_path)


# In[ ]:




