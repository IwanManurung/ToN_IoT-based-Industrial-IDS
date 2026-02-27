from . import iids_anomaly_core
from base import util
from base import params
from typing import List, Literal, Dict, Union

# 
class ICS_IIDS_Anomaly(iids_anomaly_core):
    '''
        Docstring for ICS_IIDS
        
        :param learning_models: inherit from MOA learning models. Can be set to only one or three models
        :param normalized: Bool
        :param scaler_model: Standardization model. Currently only support sklearn StandardScaler or MinMax
        :param window_size: Data windowing size
        :param random_seed: Use for both MOA learners and sampling data set (only if run_audited_version is True)
        :param run_audited_version: If True, then IIDS only review as much as sample_size of the dataset
        :param sample_size: Number of the instances to be reviewed (only if run_audited_version is True)
        '''
    def __init__(self,
                 anomaly_models: List[str] =['Autoencoder', 'HalfSpaceTrees', 'AdaptiveIsolationForest'],
                 proba_threshold: float =.6,
                #  low_memory: bool =False,
                 normalized: bool =True,
                 scaler_model: Literal['StandardScaler', 'MinMaxScaler']='StandardScaler',
                 window_size: int =60,
                 random_seed: int =1,
                 run_audited_version: bool =True,
                 audited_size: float =0.1) -> None:
        
        assert len(set(anomaly_models)) == 3, "Learning models should be 3 unique models"
        assert window_size > 0
        assert random_seed > 0
        
        configs = {}
        configs.update({"anomaly_models": anomaly_models})
        configs.update({"proba_threshold": proba_threshold})
        # configs.update({"low_memory": low_memory})
        configs.update({"normalized": normalized})
        configs.update({"scaler_model": scaler_model})
        configs.update({"window_size": window_size})
        configs.update({"random_seed": random_seed})
        configs.update({"run_audited_version": run_audited_version})
        configs.update({"audited_size": audited_size})
        self.__iids_configs = configs
    
    def print_configs(self):
        util.print_dict(dicts=self.__iids_configs,
                        startswith='IIDS Configs',
                        sort_value=False)
    
    def Edge_Layer_IIDS(self,
                        base_device: Literal['fridge', 'garage_door', 'modbus', 'motion_light', 'thermostat', 'weather'],
                        all_in_fusion: bool= True,
                        device_in_fusion: List[Literal['fridge', 'garage_door', 'modbus', 'motion_light', 'thermostat', 'weather', None]] =[],
                        prediction_class: str ='normal') -> None:
        assert base_device.lower() in params.edge_device_list, 'No base_device name found in edge device list'
        base_device = base_device.lower()
        self.__iids_configs.update({"base_device": base_device})
        self.__iids_configs.update({"all_in_fusion": all_in_fusion})
        self.__iids_configs.update({"device_in_fusion": device_in_fusion})
        self.__iids_configs.update({"layer": params.edge_layer})
        super().__init__(iids_configs=self.__iids_configs)
        iids_anomaly_core.ocl_pipeline(self, prediction_class=prediction_class)

    def Fog_Layer_IIDS(self,
                       base_device: Literal['fridge', 'garage_door', 'modbus', 'motion_light', 'thermostat', 'weather'],
                       prediction_class='normal') -> None:
        assert base_device.lower() in params.edge_device_list, 'No base_device name found in edge device list'
        base_device = base_device.lower()

        self.__iids_configs.update({"base_device": base_device})
        self.__iids_configs.update({"device_in_fusion": []})
        self.__iids_configs.update({"layer": params.fog_layer})
        super().__init__(iids_configs=self.__iids_configs)
        iids_anomaly_core.ocl_pipeline(self, prediction_class=prediction_class)