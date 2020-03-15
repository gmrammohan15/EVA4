import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import copy
from torchsummary import summary
from torchvision import datasets, transforms
from tqdm import tqdm
import network


def train_model(model, device, train_loader, optimizer, epoch, train_losses,train_acc, LAMBDA = 0.0001):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0

  criterion = nn.CrossEntropyLoss().to(device)

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
    l1_reg_loss = 0
    for param in model.parameters():
      l1_reg_loss += torch.sum(abs(param))

    classify_loss = criterion(y_pred, target)

    #loss = F.nll_loss(y_pred, target)

    loss = classify_loss + LAMBDA * l1_reg_loss

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

def test_model(model, device, test_loader, test_losses,test_acc):
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


def show_summary(model,input_size = (1, 28, 28)):
    summary(model.m_model, input_size)

def run_model(model, device, LAMBDA = 0.0001, EPOCHS = 40):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train_model(model.m_model, device, model.m_train_loader, model.m_optimizer, epoch,model.m_train_losses,model.m_train_acc,LAMBDA)
        test_model(model.m_model, device, model.m_test_loader,model.m_test_losses,model.m_test_acc)