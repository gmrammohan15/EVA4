import torch
from tqdm import tqdm
from torch.autograd import Variable


class Train:
  def __init__(self, model, dataloader, optimizer, criterion, schedular=None, l1_lambda=0):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.schedular = schedular
    self.l1_lambda = l1_lambda
    self.criterion = criterion


  def run(self, epoch):
    self.model.train()
    train_acc = 0.0
    train_loss = 0.0
    pbar = tqdm(self.dataloader)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(self.model.device), labels.to(self.model.device)
        correct = 0
        processed = 0
        # Clear all accumulated gradients
        self.optimizer.zero_grad()
        # Predict classes using images from the test set
        outputs = self.model(images)
        # Compute the loss based on the predictions and actual labels
        loss = self.criterion(outputs, labels)

        if self.l1_lambda > 0:
            reg_loss = 0.0
            for param in self.model.parameters():
                reg_loss += torch.sum(param.abs())
            loss += self.l1_lambda * reg_loss
        # Backpropagate the loss
        loss.backward()

        # Adjust parameters according to the computed gradients
        self.optimizer.step()
	
        y_pred = self.model(images)
        prediction = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += prediction.eq(labels.view_as(prediction)).sum().item()
        processed += len(images)

        pbar.set_description(desc= f'Epoch= {epoch} Loss={loss.item()} Batch_id={i} Accuracy={100*correct/processed:0.2f}')
        pbar.update(1)


    train_acc = 100*correct/processed

    return train_acc

class Test:
  def __init__(self, model, dataloader):
    self.model = model
    self.dataloader = dataloader

  def run(self, epoch):
    self.model.eval()
    test_acc = 0.0
    correct = 0
    total  = 0
    with torch.no_grad():
        test_pbar = tqdm(self.dataloader)
        for i, (images, labels) in enumerate(test_pbar):
            images, labels = images.to(self.model.device), labels.to(self.model.device)

            # Predict classes using images from the test set
            outputs = self.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        #Compute the average acc and loss over all 10000 test images
    test_acc = (100 * correct / total) 

    return test_acc


class ModelTrainer:
  def __init__(self, model, optimizer, train_loader, test_loader, criterion, schedular=None, batch_schedular=False, l1_lambda = 0):
    self.model = model
    self.schedular = schedular
    self.batch_schedular = batch_schedular
    self.optimizer = optimizer
    self.criterion = criterion
    self.train = Train(model, train_loader, optimizer, self.criterion, self.schedular if self.schedular and self.batch_schedular else None, l1_lambda)
    self.test = Test(model, test_loader)

  def run(self, epochs=10):
    for epoch in range(epochs):
        train_acc = self.train.run(epoch)
        test_acc = self.test.run(epoch)
        if self.schedular and not self.batch_schedular:
            self.schedular.step()
        print("Epoch {}, Train Accuracy: {} , Test Accuracy: {}".format(epoch, train_acc, test_acc))

