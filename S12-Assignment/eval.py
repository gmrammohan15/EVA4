import torch
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
 

class Train:
  def __init__(self, model, dataloader, optimizer, criterion, onecycle, schedular=None, l1_lambda=0):
    self.model = model
    self.dataloader = dataloader
    self.optimizer = optimizer
    self.schedular = schedular
    self.l1_lambda = l1_lambda
    self.criterion = criterion
    self.onecycle = onecycle


  # def update_lr(self, lr):
  #   for g in self.optimizer.param_groups:
  #       g['lr'] = lr


  # def update_mom(self, mom):
  #   for g in self.optimizer.param_groups:
  #       g['momentum'] = mom

  def run(self, epoch):
    self.model.train()
    train_acc = 0.0
    train_loss = 0.0
    pbar = tqdm(self.dataloader)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(self.model.device), labels.to(self.model.device)
        correct = 0
        processed = 0

        #Oncecycle
        lr, mom = self.onecycle.calc()

        #Update lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        #update momentum
        for g in self.optimizer.param_groups:
            g['momentum'] = mom    

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

    return train_acc, loss

class Test:
  def __init__(self, model, dataloader, criterion):
    self.model = model
    self.dataloader = dataloader
    self.criterion = criterion

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
            loss = self.criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        #Compute the average acc and loss over all 10000 test images
    test_acc = (100 * correct / total) 

    return test_acc, loss.cpu().numpy()


class ModelTrainer:
  def __init__(self, model, optimizer, train_loader, test_loader, criterion, onecycle, schedular=None, batch_schedular=False, l1_lambda = 0):
    self.model = model
    self.schedular = schedular
    self.batch_schedular = batch_schedular
    self.optimizer = optimizer
    self.criterion = criterion
    self.onecycle = onecycle
    self.train = Train(model, train_loader, optimizer, self.criterion, self.onecycle, self.schedular if self.schedular and self.batch_schedular else None, l1_lambda)
    self.test = Test(model, test_loader, self.criterion)
    self.test_acc = []
    self.train_acc = []
    self.train_loss = []
    self.test_loss = []

  def run(self, epochs=10):
    for epoch in range(epochs):
        train_acc, train_loss = self.train.run(epoch)
        test_acc, test_loss = self.test.run(epoch)

        self.train_acc.append(train_acc)
        self.train_loss.append(train_loss)
        self.test_acc.append(test_acc)
        self.test_loss.append(test_loss)

        if self.schedular and not self.batch_schedular:
            self.schedular.step(test_loss)
        print("Epoch {}, Train Accuracy: {} , Test Accuracy: {}".format(epoch, train_acc, test_acc))

  def plot_graph(self):
    fig, axs = plt.subplots(2,2,figsize=(10,10))
    axs[0,0].plot(self.train_acc)
    axs[0,0].set_title("Training Accuracy")
    axs[0,0].set_xlabel("Batch")
    axs[0,0].set_ylabel("Accuracy")
    axs[0,1].plot(self.test_acc) 
    axs[0,1].set_title("Test Accuracy")
    axs[0,1].set_xlabel("Batch")
    axs[0,1].set_ylabel("Accuracy")
    axs[1,0].plot(self.train_loss)
    axs[1,0].set_title("Training Loss")
    axs[1,0].set_xlabel("Batch")
    axs[1,0].set_ylabel("Loss")
    axs[1,1].plot(self.test_loss) 
    axs[1,1].set_title("Test Loss")
    axs[1,1].set_xlabel("Batch")
    axs[1,1].set_ylabel("Loss")
    