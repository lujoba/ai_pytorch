from ai_pytorch.utils.helper_functions import HelperFunctions
import time
import copy
import torch


class ModelTraining(object):
    """
    Support function for model training.

    Args:
      model: Model to be trained
      criterion: Optimization criterion (loss)
      optimizer: Optimizer to use for training
      scheduler: Instance of ``torch.optim.lr_scheduler``
      num_epochs: Number of epochs
      device: Device to run the training on. Must be 'cpu' or 'cuda'
    """
    def __init__(self, model, criterion, optimizer, scheduler, num_epochs=25, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device

    def __call__(self, dataloaders):
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        counter = []
        loss_history = []

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['training', 'testing']:
                if phase == 'training':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                # running_loss = 0.0
                # running_corrects = 0

                # Iterate over data.
                for i, data in enumerate(dataloaders[phase], 0):
                    img0, img1, label = data
                    img0, img1, label = img0.to(self.device), img1.to(self.device), label.to(self.device)
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'training'):
                        output1, output2 = self.model(img0, img1)
                        loss_contrastive = self.criterion(output1, output2, label)
                        # backward + optimize only if in training phase
                        if phase == 'training':
                            loss_contrastive.backward()
                            self.optimizer.step()
                if phase == 'training':
                    self.scheduler.step()

                print('{} Loss: {:.4f} '.format(
                    phase, loss_contrastive))
                if phase == 'testing' and loss_contrastive < best_loss:
                    best_loss = loss_contrastive
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                    counter.append(epoch)
                    loss_history.append(loss_contrastive.item())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best test Loss: {:4f}'.format(best_loss))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        HelperFunctions.show_plot(counter, loss_history)

        return self.model
