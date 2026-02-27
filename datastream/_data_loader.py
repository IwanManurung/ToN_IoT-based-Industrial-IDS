import numpy as np
import pandas as pd
from base import params
from typing import List, Literal

class data_loader:
    @staticmethod
    def load_feature(device: str):
        assert device.lower() in params.csv_sources.keys(), "no device id found in data sources"
        device = device.lower()
        src = params.csv_sources.get(device)
        return pd.read_csv(src, low_memory=False)
    
    @staticmethod
    def load_groundtruth(device: str):
        assert device.lower() in params.csv_sources.keys(), "no device id found in data sources"
        device = device.lower()
        src = params.csv_sources.get('edge_groundtruth')
        colname = [params.ts_label, f'y_{device}']
        gt = pd.read_csv(src, low_memory=False)
        gt = gt[colname].copy()
        gt = gt.rename(columns={f'y_{device}': params.target_label})
        return gt
    
    @staticmethod
    def load_edge_with_groundtruth(device: str):
        feature = data_loader.load_feature(device=device)
        target = data_loader.load_groundtruth(device=device)
        data = pd.merge(feature, target, on=params.ts_label, how='outer')
        data[params.target_label] = data[params.target_label].fillna(params.normal_l1label)
        data = data.dropna().drop_duplicates().sort_values(params.ts_label).reset_index(drop=True)
        return data
    
    @staticmethod
    def load_npz(device: str,
                 layer: Literal['edge', 'fog']):
        assert layer.lower() in params.npz_sources.keys()
        layer = layer.lower()
        assert device.lower() in params.npz_sources.get(layer).keys()
        device = device.lower()
        src = params.npz_sources.get(layer).get(device)
        loaded = np.load(src, allow_pickle=True)
        feature = loaded.get('feature')
        target = loaded.get('target').tolist()
        header = loaded.get('header').tolist()
        return (feature, target, header)
    
class _prep_ToN_IoT:
    def __init__(self):
        pass
    
    @staticmethod
    def __downloader():
        pass

    @staticmethod
    def __preprocessing_fridge():
        pass

    @staticmethod
    def __preprocessing_garage_door():
        pass

    @staticmethod
    def __preprocessing_modbus():
        pass

    @staticmethod
    def __preprocessing_motion_light():
        pass

    @staticmethod
    def __preprocessing_thermostat():
        pass

    @staticmethod
    def __preprocessing_weather():
        pass

    @staticmethod
    def __preprocessing_linux_memory():
        pass

    @staticmethod
    def __preprocessing_linux_disks():
        pass

    @staticmethod
    def __preprocessing_linux_processes():
        pass

    @staticmethod
    def __preprocessing_windows7():
        pass

    @staticmethod
    def __preprocessing_windows10():
        pass

    @staticmethod
    def __make_fog_datasets():
        pass

    @staticmethod
    def __make_edge_datasets():
        pass

class ToN_IoT:
    def __init__(self):
        pass

    def Edge_Fusion_Fridge(self):

        self.feature
        self.target
        self.header
        pass

    def Edge_Fusion_Garage_Door(self):
        self.feature
        self.target
        self.header
        pass

    def Edge_Fusion_Modbus(self):
        self.feature
        self.target
        self.header
        pass

    def Edge_Fusion_Motion_Light(self):
        self.feature
        self.target
        self.header
        pass

    def Edge_Fusion_Thermostat(self):
        self.feature
        self.target
        self.header
        pass

    def Edge_Fusion_Weather(self):
        self.feature
        self.target
        self.header
        pass

    def Fog_Fusion_Fridge(self):
        self.feature
        self.target
        self.header
        pass

    def Fog_Fusion_Garage_Door(self):
        self.feature
        self.target
        self.header
        pass

    def Fog_Fusion_Modbus(self):
        self.feature
        self.target
        self.header
        pass

    def Fog_Fusion_Motion_Light(self):
        self.feature
        self.target
        self.header
        pass

    def Fog_Fusion_Thermostat(self):
        self.feature
        self.target
        self.header
        pass

    def Fog_Fusion_Weather(self):
        self.feature
        self.target
        self.header
        pass

    class __as_stream:
        def to_stream(self, label_on):
            stream = 'xx'
            return stream

    