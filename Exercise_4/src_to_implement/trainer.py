import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm # tqdm is to get a progress bar


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    '''
    x is images
    y is labels
    step means the process for each image
    '''
    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()

        # -propagate through the network
        output = self._model(x)

        # -calculate the loss
        loss = self._crit(output, y)

        # -compute gradient by backward propagation
        loss.backward()

        # -update weights
        self._optim.step()
        
        # -return the loss
        return loss
        
    
    def val_test_step(self, x, y):
        # predict
        output = self._model(x)

        # propagate through the network and calculate the loss and predictions
        loss = self._crit(output, y).item()
        # get the index of max probability as prediction
        _, pred = t.max(output, 1)
        
        # return the loss and the predictions
        return loss, pred
        
    
    def train_epoch(self):
        loss = 0.0
        
        # set training mode
        self._model.train() 
        
        # iterate through the training set
        for (data, label) in tqdm(enumerate(self._train_dl)):
            # transfer the batch to "cuda()" -> the gpu if a gpu is given
            data.to(t.device('cuda'))
            label.to(t.device('cuda'))
            
            # perform a training step
            loss += self.train_step(data, label).item()

        # calculate the average loss for the epoch and return it
        return loss / len(self._train_dl)
    
    def val_test(self):
        test_loss = 0.0

        label_list = []
        pred_list = []
        
        # set eval mode
        self._model.eval()
        print("Test Process")
        
        # disable gradient computation
        with t.no_grad():
            # iterate through the validation set
            for (data, label) in tqdm(self._val_test_dl):
                # transfer the batch to the gpu if given
                data.to(t.device('cuda'))
                label.to(t.device('cuda'))
                
                # perform a validation step
                loss, pred = self.val_test_step(data, label)
                test_loss += loss
                
                # save the predictions and the labels for each batch
                label_list.append(label)
                pred_list.append(pred)

        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        avg_metric = f1_score(label_list, pred_list)
        avg_loss = test_loss / len(self._val_test_dl)
        
        # return the loss and print the calculated metrics
        return avg_loss, avg_metric
        
    
    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        vali_losses = []
        counter = 0
        
        while True:
            
            # stop by epoch number
            if counter >= epochs:
                break

            if counter >= self._early_stopping_patience:
                # train for a epoch and then calculate the loss and metrics on the validation set
                train_loss = self.train_epoch()
                vali_loss, vali_metric = self.val_test()
                
                # append the losses to the respective lists
                train_losses.append(train_loss)
                vali_losses.append(vali_loss)
                
                # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
                self.save_checkpoint(counter)
                # check whether early stopping should be performed using the early stopping criterion and stop if so
                if vali_losses[-1] - vali_losses[-2] >= 0:
                    print("Early Stopping")
                    break
                else:
                    counter += 1
                    continue
            else:
                train_loss = self.train_epoch()
                print("Train Loss for %d th Epoch is %f" %(counter, train_loss))
                counter += 1
            
            res = []
            res.append(train_losses)
            res.append(vali_losses)
            # return the losses for both training and validation
            return res
                    
        
        
        