from IoTAnomalyDetectorBase import IoTAnomalyDetectorBase
from IoTAnomalyDetectorLSTMEncDecNet import IoTAnomalyDetectorLSTMEncDecNet
from utils import *


class IoTAnomalyDetectorLSTMEncDec(IoTAnomalyDetectorBase):
    def __init__(self, data, seq_len, loss, optimizer, learning_rate, epochs, batch_size, train_val_split, is_cli):
        IoTAnomalyDetectorBase.__init__(self, data, IoTAnomalyDetectorLSTMEncDecNet(), seq_len, loss, optimizer,
                                        learning_rate, epochs, batch_size, train_val_split, is_cli)
        self.tr_star = None
        self.seq_len = seq_len

    def seq_generator(self, data):
        mean_data = np.mean(data, axis=0)
        # pad the beginning of the data with mean rows in order to minimize the error
        # on first rows while using sequence
        prefix_padding = np.asarray([mean_data for _ in range(self.seq_len - 1)])
        padded_data = np.vstack((prefix_padding, data))
        # mse_per_sample_in_seq of shape (n_seq, seq_len)
        for i in range(data.shape[0]):
            seq = padded_data[i:i + self.seq_len]
            yield seq

    def get_tr_star(self):
        """
        tr star is a vector of length seq_len

        :return:
        """
        if self.tr_star is not None:
            return self.tr_star
        # preds of shape (n_seq, seq_len, n_features)
        preds = self.predict(self.x_mat_val)

        mse_per_sample_in_seq = []
        for i, seq in enumerate(self.seq_generator(self.x_mat_val)):
            mse_per_sample_in_seq.append(calc_mse(seq, preds[i], axis=1))

        mse_per_sample_in_seq = np.asarray(mse_per_sample_in_seq)
        return mse_per_sample_in_seq.mean(axis=0) + mse_per_sample_in_seq.std(axis=0)

    def _learn_post_training_parameters(self):
        self.tr_star = self.get_tr_star()

    def detect_anomalies(self, x_mat):
        """
        x_mat of shape(seq_len, n_seq, n_features)
        :param x_mat:
        :return:
        """
        x_mat_scaled = self.scaler.transform(x_mat)
        # preds of shape (n_seq, seq_len, n_features)
        preds = self.predict(x_mat_scaled)

        # is_anomaly of shape (seq_len, n_seq)
        is_anomaly = []
        for i, seq in enumerate(self.seq_generator(x_mat)):
            is_anomaly.append(calc_mse(seq, preds[i], axis=1) > self.get_tr_star())

        is_anomaly_majority_vote = np.sum(is_anomaly, axis=1)
        is_anomaly_majority_vote = is_anomaly_majority_vote > (self.seq_len / 2)
        return is_anomaly_majority_vote, None

    def get_post_training_parameters(self):
        return {"tr_star": self.get_tr_star()}

    def __repr__(self):
        return "model: {}\n" \
               "tr*:   {}\n".format(self.net, self.get_tr_star())

