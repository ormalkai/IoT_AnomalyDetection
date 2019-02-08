from abc import ABC, abstractmethod
from torch import nn, optim
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm_notebook, tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


class IoTAnomalyDetectorBase(ABC):
    def __init__(self, data, net, seq_len, loss, optimizer, learning_rate, epochs, batch_size, train_val_split, is_cli,
                 normalize="min-max"):
        """
        TODO ORM doc
        :param data:
        :param net:
        :param learning_rate:
        :param epochs:
        :param batch_size:
        :param train_val_split:
        """
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = net
        self.is_cli = is_cli
        if train_val_split <= 0 or train_val_split >= 1:
            raise Exception("train_val_split must be in range (0, 1)")
        x_mat_train = []
        x_mat_val = []
        for iot in data:
            train_size = int(len(iot) * train_val_split)
            iot_train = iot[:train_size]
            iot_val = iot[train_size:]
            x_mat_train.extend(iot_train)
            x_mat_val.extend(iot_val)
            if normalize == "norm":
                self.scaler = StandardScaler(copy=False)
            elif normalize == "min-max":
                self.scaler = MinMaxScaler(copy=False)
            else:
                raise Exception("Unexpected argument normalize={}".format(normalize))
            self.scaler.fit(iot_train)
            self.scaler.transform(iot_train)
            self.scaler.transform(iot_val)

        self.x_mat_train = np.asarray(x_mat_train)
        self.y_train = np.asarray(x_mat_train)
        self.x_mat_val = np.asarray(x_mat_val)
        self.y_val = np.asarray(x_mat_val)

        if loss == "MSE":
            self.loss = nn.MSELoss()
        else:
            raise Exception("Unsupported loss {}".format(loss))
        if optimizer == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), self.learning_rate)
        elif optimizer == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), self.learning_rate)
        else:
            raise Exception("Unsupported optimizer {}".format(optim))

    @staticmethod
    def batch_generator(x_mat, y, batch_size, seq_len):
        if seq_len == 1:
            # Currently no permutation since the chronological order is important!
            permutation = list(range(x_mat.size()[0]))
            for i in range(0, x_mat.size()[0], batch_size):
                indices = permutation[i:i + batch_size]
                input_x = x_mat[indices]
                expected_y = y[indices]
                if torch.cuda.is_available():
                    yield Variable(torch.cuda.FloatTensor(input_x)), Variable(torch.cuda.FloatTensor(expected_y))
                else:
                    yield Variable(torch.FloatTensor(input_x)), Variable(torch.FloatTensor(expected_y))
        else:
            mean_data_x = np.mean(x_mat, axis=0)
            mean_data_y = np.mean(y, axis=0)
            # pad the beginning of the data with mean rows in order to minimize the error
            # on first rows while using sequence
            prefix_padding_x = np.asarray([mean_data_x for _ in range(seq_len - 1)])
            prefix_padding_y = np.asarray([mean_data_y for _ in range(seq_len - 1)])
            padded_data_x = np.vstack((prefix_padding_x, x_mat))
            padded_data_y = np.vstack((prefix_padding_y, y))
            seq_data = []
            seq_y = []
            for i in range(len(padded_data_x) - seq_len + 1):
                seq_data.append(padded_data_x[i:i + seq_len, :])
                seq_y.append(padded_data_y[i:i + seq_len, :])
                if len(seq_data) == batch_size:
                    if torch.cuda.is_available():
                        yield Variable(torch.cuda.FloatTensor(seq_data)), Variable(torch.cuda.FloatTensor(seq_y))
                    else:
                        yield Variable(torch.FloatTensor(seq_data)), Variable(torch.FloatTensor(seq_y))
                    seq_data = []
                    seq_y = []
            if len(seq_data) > 0:  # handle data which is not multiply of batch size
                if torch.cuda.is_available():
                    yield Variable(torch.cuda.FloatTensor(seq_data)), Variable(torch.cuda.FloatTensor(seq_y))
                else:
                    yield Variable(torch.FloatTensor(seq_data)), Variable(torch.FloatTensor(seq_y))

    def run_epoch(self, x_mat, y, is_train, batch_size, seq_len):
        epoch_loss = 0.0
        # iterate over the data in batches
        epoch_losses = []
        epoch_outputs = []
        if batch_size == -1:
            batch_size = x_mat.shape[0]
        curr_tqdm = tqdm if self.is_cli else tqdm_notebook
        for input_x, expected_y in curr_tqdm(IoTAnomalyDetectorBase.batch_generator(x_mat, y, batch_size, seq_len),
                                                 desc='batch', leave=False):

            # zero optimzer gradients
            # I n PyTorch, we need to set the gradients to zero before starting to do backpropragation because
            # PyTorch accumulates the gradients on subsequent backward passes.
            # This is convenient while training RNNs.
            # So, the default action is to accumulate the gradients on every loss.backward() call.
            #
            # Because of this, when you start your training loop, ideally you should zero out the gradients so
            # that you do the parameter update correctly.
            # Else the gradient would point in some other direction
            if is_train:
                self.net.train()
                self.optimizer.zero_grad()
            else:
                self.net.eval()

            # feed forward
            output = self.net.forward(input_x)
            # calc loss
            loss = self.loss(output, expected_y)

            # backprop
            if is_train:
                loss.backward()
                self.optimizer.step()

            # update statistics
            # loss.item() returns the scalar value of the loss
            # loss is averaged by batch size
            epoch_losses.append(loss.item())  # for calculating tr* while running epoch with batch_size=1
            epoch_loss += input_x.shape[0] * loss.item()
            if torch.cuda.is_available():
                epoch_outputs.extend(output.clone().detach().cpu().numpy())
            else:
                epoch_outputs.extend(output.clone().detach().numpy())
        return epoch_loss, epoch_losses, np.asarray(epoch_outputs)

    def fit(self, plot_name):
        self.net.train()
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        curr_tqdm = tqdm if self.is_cli else tqdm_notebook
        for epoch in curr_tqdm(range(self.epochs), desc='epoch'):
            train_running_loss, _, _ = self.run_epoch(self.x_mat_train, self.y_train, is_train=True,
                                                      batch_size=self.batch_size, seq_len=self.seq_len)
            val_running_loss, _, _ = self.run_epoch(self.x_mat_val, self.y_val, is_train=False,
                                                    batch_size=self.batch_size, seq_len=self.seq_len)

            # print statistics
            train_running_loss /= len(self.x_mat_train)
            val_running_loss /= len(self.x_mat_val)
            if epoch % 50 == 0:
                print("Epoch {}, train avg loss {:.6f}, val avg loss {:.6f}".format(epoch, train_running_loss, val_running_loss))
            train_loss_per_epoch.append(train_running_loss)
            val_loss_per_epoch.append(val_running_loss)
        # plot learning curve
        # summarize history for loss
        plt.plot(train_loss_per_epoch, linewidth=3)
        plt.plot(val_loss_per_epoch, linewidth=3)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper right')
        if plot_name is not None:
            plt.savefig(plot_name)
        else:
            plt.show()

    def predict(self, x_mat):
        _, _, predictions = self.run_epoch(x_mat, x_mat, is_train=False, batch_size=self.batch_size,
                                           seq_len=self.seq_len)
        return predictions

    # def predict(self, x_mat):
    #     self.net.eval()
    #     if torch.cuda.is_available():
    #         x_mat = torch.cuda.FloatTensor(x_mat)
    #     else:
    #         x_mat = torch.FloatTensor(x_mat)
    #     return self.net.forward(Variable(x_mat)).cpu().data.numpy()

    def learn_benign_baseline(self, model_filename, is_train, plot_name):
        if is_train:
            # train the model
            self.fit(plot_name)
            self.net.save_model_to_file(model_filename)
        else:
            self.load_model_from_file(model_filename)
        self._learn_post_training_parameters()

    @abstractmethod
    def _learn_post_training_parameters(self):
        pass

    @abstractmethod
    def detect_anomalies(self, x_mat):
        pass

    def load_model_from_file(self, model_filename):
        if model_filename is None:
            raise Exception("Must train model or load pre-trained model")
        print("Loading pre trained model from file {}".format(model_filename))
        self.net.load_model_from_file(model_filename)

    @abstractmethod
    def get_post_training_parameters(self):
        pass
