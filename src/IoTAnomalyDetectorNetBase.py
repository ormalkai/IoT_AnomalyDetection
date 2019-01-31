class IoTAnomalyDetectorModelBase:
    def forward(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

#####################################################################################


class facial:
    def __init__(self, data, labels, fc_or_conv, learning_rate=0.01, epochs=1000, batch_size=128):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        if fc_or_conv == "fc":
            self.net = facial_fc_net()
        else:  # fc_or_conv == "conv"
            self.net = facial_conv_net()
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


def load_facial_data(file_name, is_test=False, fc_or_conv="fc"):
    # load
    df = pd.read_csv(file_name)
    # drop na values, use only 2140 pictures with full data for training
    df = df.dropna()
    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    # Scale image pixel values to [0, 1]
    x_mat = np.vstack(df['Image'].values) / 255.
    x_mat = x_mat.astype(np.float32)
    if fc_or_conv == "conv":
        x_mat = x_mat.reshape(-1, 1, 96, 96)
    if is_test:  # we actually not using the test dataset in this exercise
        return x_mat, None

    # Target values are all columns except last one which contains the image
    y = df[df.columns[:-1]].values.astype(np.float32)
    # scale target coordinates to [-1, 1]
    #     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    #     y = min_max_scaler.fit_transform(y)
    y = (y - 48) / 48
    return x_mat, y


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def run(fc_or_conv):
    x_mat, y = load_facial_data("/content/drive/My Drive/לימודים/Deep Learning/שיעורים/ex1/EX1/EX1/Q3/training.zip",
                                is_test=False, fc_or_conv=fc_or_conv)

    facial_op = facial(data=x_mat, labels=y, fc_or_conv=fc_or_conv)
    train_loss_per_epoch, val_loss_per_epoch = facial_op.fit()
    # summarize history for loss
    plt.plot(train_loss_per_epoch, linewidth=3)
    plt.plot(val_loss_per_epoch, linewidth=3)
    plt.ylim(1e-3, 1e-2)
    plt.yscale("log")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.show()

    # predict on test set
    x_mat, _ = load_facial_data("/content/drive/My Drive/לימודים/Deep Learning/שיעורים/ex1/EX1/EX1/Q3/test.zip",
                                is_test=True, fc_or_conv=fc_or_conv)
    y = facial_op.predict(x_mat)
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(x_mat[i], y[i], ax)
    plt.show()
    return train_loss_per_epoch, val_loss_per_epoch


def run_q_3_2():
    return run("fc")


def run_q_3_3():
    return run("conv")
