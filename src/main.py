from NBaIoTDatasetLoader import NBaIoTDatasetLoader
from IoTAnomalyDetectorAutoEncoder import IoTAnomalyDetectorAutoEncoder
from IoTAnomalyDetectorLSTMEncDec import IoTAnomalyDetectorLSTMEncDec
from DatasetConsts import *
from Config import Config, parse_args
from ConfigConsts import *
from collections import OrderedDict
import numpy as np
import pandas as pd
import os


def build_train_dataset(dataset):
    """
    TODO doc
    :param dataset:
    :return:
    """
    train_dataset = []
    for iot_data in dataset.values():
        train_dataset.append(iot_data[BENIGN_TRAIN])
        iot_data.pop(BENIGN_TRAIN)
    return np.asarray(train_dataset)


def track_model(config, post_training_parameters, df_results):
    tracking_file = PROJECT_PATH + "tracking/" + "tracking_models2.csv"  # TODO ORM take from config
    old_tracking = pd.DataFrame()
    # if file exists load dataframe
    if os.path.isfile(tracking_file):
        old_tracking = pd.read_csv(tracking_file)
    # create new dataframe
    # concat dicts
    config_dict = vars(config)
    for iot in df_results.columns:
        iot_results = df_results[iot]
        iot_col = 'iot'
        cols = list(config_dict.keys()) + list(post_training_parameters.keys()) + [iot_col]  + list(iot_results.keys())
        tracking_line = OrderedDict()
        tracking_line = {**tracking_line, **config_dict}
        tracking_line = {**tracking_line, **post_training_parameters}
        current_iot   = {iot_col: iot}
        tracking_line = {**tracking_line, **current_iot}
        tracking_line = {**tracking_line, **iot_results}
        # convert iot list to string
        tracking_line['iots'] = '_'.join(tracking_line['iots'])
        # append results
        # old_tracking = old_tracking.append(pd.DataFrame.from_dict(tracking_line)[cols])
        old_tracking = old_tracking.append(pd.Series(tracking_line), ignore_index=True)[cols]
    # save the dataframe
    old_tracking.to_csv(tracking_file, index=False)


def main(args=None):
    # parse args
    config = Config()
    parse_args(config, args)
    # Load data
    dataset = NBaIoTDatasetLoader.load_data(config.iots, config.train_test_split)
    # Build train dataset
    train_dataset = build_train_dataset(dataset)
    # Create model
    if config.model == MODEL_AUTOENCODER:
        iot_anomaly_detector = IoTAnomalyDetectorAutoEncoder(data=train_dataset,
                                                             seq_len=config.seq_len,
                                                             loss=config.loss,
                                                             optimizer=config.optimizer,
                                                             learning_rate=config.learning_rate,
                                                             epochs=config.epochs,
                                                             batch_size=config.batch_size,
                                                             train_val_split=config.train_val_split,
                                                             is_cli=config.is_cli)
    elif config.model == MODEL_LSTM_ENC_DEC:
        iot_anomaly_detector = IoTAnomalyDetectorLSTMEncDec(data=train_dataset,
                                                            seq_len=config.seq_len,
                                                            loss=config.loss,
                                                            optimizer=config.optimizer,
                                                            learning_rate=config.learning_rate,
                                                            epochs=config.epochs,
                                                            batch_size=config.batch_size,
                                                            train_val_split=config.train_val_split,
                                                            is_cli=config.is_cli)
    else:
        raise Exception("Unknown model {}".format(config.model))

    # handle model file name
    if config.model_filename is not None:
        config.model_filename = PROJECT_PATH + MODELS_DIR + config.model_filename

    # Check if not pre-trained model
    print("training start")
    iot_anomaly_detector.learn_benign_baseline(config.model_filename, config.train,
                                               plot_name=FIGURES_PATH+config.name+".png" if config.is_cli else None,
                                               temporary_model_filename=config.temporary_model_filename)
    print(iot_anomaly_detector)

    # if need to detect anomalies -> calc TPR and FPR for each attack
    results = dict()
    if config.test:
        for iot, iot_data in dataset.items():
            iot_results = dict()
            for traffic_type, traffic_data in iot_data.items():
                if traffic_data is None:
                    iot_results[traffic_type] = None
                    continue
                is_anomaly_majority_vote, _ = iot_anomaly_detector.detect_anomalies(traffic_data)
                iot_results[traffic_type] = np.sum(is_anomaly_majority_vote) / np.size(is_anomaly_majority_vote)
            results[iot] = iot_results
    df_results = pd.DataFrame.from_dict(results)
    # track configuration, post train parameters, score
    track_model(config, iot_anomaly_detector.get_post_training_parameters(), df_results)
    return df_results


if __name__ == "__main__":
    main()

