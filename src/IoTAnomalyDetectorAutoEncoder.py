from IoTAnomalyDetectorBase import IoTAnomalyDetectorBase
from IoTAnomalyDetectorAutoEncoderNet import IoTAnomalyDetectorAutoEncoderNet
from utils import *
import numpy as np


class IoTAnomalyDetectorAutoEncoder(IoTAnomalyDetectorBase):
    def __init__(self, data, seq_len, loss, optimizer, learning_rate, epochs, batch_size, train_val_split):
        IoTAnomalyDetectorBase.__init__(self, data, IoTAnomalyDetectorAutoEncoderNet(), seq_len, loss, optimizer,
                                        learning_rate, epochs, batch_size, train_val_split)
        self.tr_star = None
        self.ws_star = None

    def get_tr_star(self):
        if self.tr_star is not None:
            return self.tr_star
        preds = self.predict(self.x_mat_val)
        mse_per_sample = calc_mse(self.x_mat_val.data.numpy(), preds)
        return mse_per_sample.mean() + mse_per_sample.std()

    def get_ws_star(self):
        if self.ws_star is not None:
            return self.ws_star
        preds = self.predict(self.x_mat_val)
        mse_per_sample = calc_mse(self.x_mat_val.data.numpy(), preds)
        is_anomaly = mse_per_sample > self.get_tr_star()

        for ws in range(1, self.x_mat_val.size()[0]):
            has_fp = False
            for i in range(ws, is_anomaly.shape[0]):
                vote = np.sum(is_anomaly[i-ws+1:i+1])
                if vote > ws / 2:  # Found False-Positive
                    has_fp = True
                    break
            if not has_fp:
                return ws
        raise Exception("Couldn't find ws star")

    def _learn_post_training_parameters(self):
        self.tr_star = self.get_tr_star()
        self.ws_star = self.get_ws_star()

    def detect_anomalies(self, x_mat):
        x_mat_scaled = self.scaler.transform(x_mat)
        preds = self.predict(x_mat_scaled)
        mse_per_sample = calc_mse(x_mat_scaled, preds)
        is_anomaly = mse_per_sample > self.get_tr_star()
        is_anomaly_majority_vote = np.zeros_like(is_anomaly)
        for i in range(is_anomaly.shape[0]):
            left = max(0, i - self.get_ws_star())
            vote = np.sum(is_anomaly[left:i+1])
            is_anomaly_majority_vote[i] = vote >= ((self.get_ws_star())//2)
        return is_anomaly_majority_vote, is_anomaly

    def get_post_training_parameters(self):
        return {"tr_star": self.get_tr_star(), "ws_star": self.get_ws_star()}

    def __repr__(self):
        return "model: {}\n" \
               "tr*:   {}\n" \
               "ws*:   {}\n".format(self.net, self.get_tr_star(), self.get_ws_star())

