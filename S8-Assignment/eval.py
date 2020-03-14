import torch

from torch.autograd import Variable


class Train:
  def __init__(self, model, dataloader, optimizer, criterion, schedular=None, l1_lambda=0):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.schedular = schedular
    self.l1_lambda = l1_lambda
    self.criterion = criterion


  def run(self):
    self.model.train()
    train_acc = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(self.dataloader):
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

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

        _, prediction = torch.max(outputs.data, 1)
        
        train_acc += torch.sum(prediction == labels.data)

    train_acc = train_acc / 50000

    return train_acc

class Test:
  def __init__(self, model, dataloader):
    self.model = model
    self.dataloader = dataloader

  def run(self):
    self.model.eval()
    test_acc = 0.0
    with torch.no_grad():
      for i, (images, labels) in enumerate(self.dataloader):
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
          images = Variable(images.cuda())
          labels = Variable(labels.cuda())

            # Predict classes using images from the test set
        outputs = self.model(images)
        _, prediction = torch.max(outputs.data, 1)
        
        test_acc += torch.sum(prediction == labels.data)

    #Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000

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
        train_acc = self.train.run()
        test_acc = self.test.run()
        if self.schedular and not self.batch_schedular:
            self.schedular.step()
        print("Epoch {}, Train Accuracy: {} , Test Accuracy: {}".format(epoch, train_acc, test_acc))

