from sklearn.model_selection import train_test_split


class IoTAnomalyDetector:
    def __init__(self, data, labels, net, learning_rate=0.01, epochs=1000, batch_size=128):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = net
        # shuffle and split train data
        if labels is not None:
            x_mat_train, x_mat_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
        else:
            x_mat_train, y_train, x_mat_val, y_val = data, None, None, None
        self.x_mat_train = Variable(torch.cuda.FloatTensor(x_mat_train))
        self.y_train = Variable(torch.cuda.FloatTensor(y_train))
        self.x_mat_val = Variable(torch.cuda.FloatTensor(x_mat_val))
        self.y_val = Variable(torch.cuda.FloatTensor(y_val))
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), self.learning_rate, nesterov=True, momentum=0.9)

    def run_epoch(self, x_mat, y, is_train):
        epoch_loss = 0.0
        permutation = torch.randperm(x_mat.size()[0])
        # iterate over the data in batches
        for i in range(0, x_mat.size()[0], self.batch_size):
            indices = permutation[i:i + self.batch_size]
            input_x = x_mat[indices]
            expected_y = y[indices]

            # zero optimzer gradients
            # I n PyTorch, we need to set the gradients to zero before starting to do backpropragation because
            # PyTorch accumulates the gradients on subsequent backward passes.
            # This is convenient while training RNNs.
            # So, the default action is to accumulate the gradients on every loss.backward() call.
            #
            # Because of this, when you start your training loop, ideally you should zero out the gradients so
            # that you do the parameter update correctly.
            # Else the gradient would point in some other directi
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
            epoch_loss += input_x.size()[0] * loss.item()
        return epoch_loss

    def fit(self):
        train_loss_per_epoch = []
        val_loss_per_epoch = []
        for epoch in range(self.epochs):
            train_running_loss = self.run_epoch(self.x_mat_train, self.y_train, is_train=True)
            val_running_loss = self.run_epoch(self.x_mat_val, self.y_val, is_train=False)

            # print statistics
            train_running_loss /= self.x_mat_train.shape[0]
            val_running_loss /= self.x_mat_val.shape[0]
            if epoch % 50 == 0:
                pass
            #                 print("Epoch {}, train avg loss {:.6f}, val avg loss {:.6f}".format(epoch, train_running_loss, val_running_loss))
            train_loss_per_epoch.append(train_running_loss)
            val_loss_per_epoch.append(val_running_loss)
        return train_loss_per_epoch, val_loss_per_epoch

    def predict(self, x_mat):
        return self.net.forward(Variable(torch.cuda.FloatTensor(x_mat))).cpu().data.numpy()