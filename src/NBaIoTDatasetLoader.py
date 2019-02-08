import pandas as pd
import numpy as np
from DatasetConsts import *
from ConfigConsts import *


class NBaIoTDatasetLoader:
    @staticmethod
    def load_data(iots, train_test_split):
        dataset = {}
        for iot in iots:
            if iot == DANMINI_DOORBELL:
                dataset[iot] = NBaIoTDatasetLoader.__load_damini_doorbell_data(train_test_split)
            elif iot == ENNIO_DOORBELL:
                dataset[iot] = NBaIoTDatasetLoader.__load_ennio_doorbell_data(train_test_split)
            elif iot == ECOBEE_THERMOSTAT:
                dataset[iot] = NBaIoTDatasetLoader.__load_ecobee_thermostat_data(train_test_split)
            elif iot == PHILIPS_BABY_MONITOR:
                dataset[iot] = NBaIoTDatasetLoader.__load_philips_baby_monitor_data(train_test_split)
            elif iot == PROV_737_SECURITY_CAM:
                dataset[iot] = NBaIoTDatasetLoader.__load_provision_737_security_camera_data(train_test_split)
            elif iot == PROV_838_SECURITY_CAM:
                dataset[iot] = NBaIoTDatasetLoader.__load_provision_838_security_camera_data(train_test_split)
            elif iot == SAMSUNG_WEB_CAM:
                dataset[iot] = NBaIoTDatasetLoader.__load_samsung_web_cam_data(train_test_split)
            elif iot == SIMPLE_HOME_1002_SECURITY_CAM:
                dataset[iot] = NBaIoTDatasetLoader.__load_simple_home_1002_security_camera_data(train_test_split)
            elif iot == SIMPLE_HOME_1003_SECURITY_CAM:
                dataset[iot] = NBaIoTDatasetLoader.__load_simple_home_1003_security_camera_data(train_test_split)
            else:
                raise Exception("Invalid iot {}".format(iot))

        if not dataset:
            raise Exception("Dataset is empty, list of iot devices {}".format(iots))

        return dataset

    @staticmethod
    def __load_data(data_dir, benign, gafgyt, mirai, train_test_split):
        df_benign = pd.read_csv(data_dir + BENIGN_TRAFFIC_FILE).dropna().values if benign else None
        df_benign_train = None
        df_benign_test = None
        if df_benign is not None:
            train_size = int(len(df_benign) * train_test_split)
            df_benign_train = df_benign[:train_size]
            df_benign_test = df_benign[train_size:]
        df_gafgyt_combo = pd.read_csv(data_dir + GAFGYT_COMBO_FILE).dropna().values if gafgyt else None
        df_gafgyt_junk = pd.read_csv(data_dir + GAFGYT_JUNK_FILE).dropna().values if gafgyt else None
        df_gafgyt_scan = pd.read_csv(data_dir + GAFGYT_SCAN_FILE).dropna().values if gafgyt else None
        df_gafgyt_tcp = pd.read_csv(data_dir + GAFGYT_TCP_FILE).dropna().values if gafgyt else None
        df_gafgyt_udp = pd.read_csv(data_dir + GAFGYT_UDP_FILE).dropna().values if gafgyt else None
        df_mirai_ack = pd.read_csv(data_dir + MIRAI_ACK_FILE).dropna().values if mirai else None
        df_mirai_scan = pd.read_csv(data_dir + MIRAI_SCAN_FILE).dropna().values if mirai else None
        df_mirai_syn = pd.read_csv(data_dir + MIRAI_SYN_FILE).dropna().values if mirai else None
        df_mirai_udp = pd.read_csv(data_dir + MIRAI_UDP_FILE).dropna().values if mirai else None
        df_mirai_udpplain = pd.read_csv(data_dir + MIRAI_UDPPLAIN_FILE).dropna().values if mirai else None

        dfs = {
            BENIGN_TRAIN: df_benign_train,
            BENIGN_TEST: df_benign_test,
            GAFGYT_COMBO: df_gafgyt_combo,
            GAFGYT_JUNK: df_gafgyt_junk,
            GAFGYT_SCAN: df_gafgyt_scan,
            GAFGYT_TCP: df_gafgyt_tcp,
            GAFGYT_UDP: df_gafgyt_udp,
            MIRAI_ACK: df_mirai_ack,
            MIRAI_SCAN: df_mirai_scan,
            MIRAI_SYN: df_mirai_syn,
            MIRAI_UDP: df_mirai_udp,
            MIRAI_UDPPLAIN: df_mirai_udpplain,
        }
        return dfs

    @staticmethod
    def __load_damini_doorbell_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(DANMINI_DOORBELL_DIR, benign=True, gafgyt=True, mirai=True,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_ennio_doorbell_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(ENNIO_DOORBELL_DIR, benign=True, gafgyt=True, mirai=False,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_ecobee_thermostat_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(ECOBEE_THERMOSTAT_DIR, benign=True, gafgyt=True, mirai=True,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_philips_baby_monitor_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(PHILIPS_BABY_MONITOR_DIR, benign=True, gafgyt=True, mirai=True,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_provision_737_security_camera_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(PROV_737_SECURITY_CAM_DIR, benign=True, gafgyt=True, mirai=True,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_provision_838_security_camera_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(PROV_838_SECURITY_CAM_DIR, benign=True, gafgyt=True, mirai=True,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_samsung_web_cam_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(SAMSUNG_WEB_CAM_DIR, benign=True, gafgyt=True, mirai=False,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_simple_home_1002_security_camera_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(SH_1002_SECURITY_CAM_DIR, benign=True, gafgyt=True, mirai=True,
                                               train_test_split=train_test_split)

    @staticmethod
    def __load_simple_home_1003_security_camera_data(train_test_split):
        return NBaIoTDatasetLoader.__load_data(SH_1003_SECURITY_CAM_DIR, benign=True, gafgyt=True, mirai=True,
                                               train_test_split=train_test_split)


