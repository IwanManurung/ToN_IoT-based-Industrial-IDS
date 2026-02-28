from . import iids_delay_label_core
from base import params, util
import json
from typing import List, Literal, Dict, Union
# 
class ICS_IIDS_Delay_Label(iids_delay_label_core):

    def __init__(self,
                 anomaly_models: List[str] =['HalfSpaceTrees', 'Autoencoder', 'AdaptiveIsolationForest'],
                 classifier_models: List[str] =['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree'],
                 delayed_label: int=1_000,
                 normalized: bool =True,
                 similarity_threshold: float =.95,
                 scaler_model: Literal['StandardScaler', 'MinMaxScaler']='StandardScaler',
                 window_size: int =600,
                 random_seed: int =1,
                 run_audited_version: bool =True,
                 audited_size: float =0.2) -> None:

        assert len(set(classifier_models)) == 3, "Learning models should be 3 unique models"
        assert window_size > 0
        assert random_seed > 0
        
        configs = {}
        configs.update({"anomaly_models": anomaly_models})
        configs.update({"classifier_models": classifier_models})
        configs.update({"delayed_label": delayed_label})
        configs.update({"normalized": normalized})
        configs.update({"similarity_threshold": similarity_threshold})
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
                        base_device: str,
                        all_in_fusion: bool=True,
                        device_in_fusion: List[str] =[],
                        prediction_class='normal') -> None:
        assert base_device.lower() in params.edge_device_list, 'No base_device name found in edge device list'
        base_device = base_device.lower()
        self.__iids_configs.update({"base_device": base_device})
        self.__iids_configs.update({"all_in_fusion": all_in_fusion})
        self.__iids_configs.update({"device_in_fusion": device_in_fusion})
        self.__iids_configs.update({"layer": params.edge_layer})
        super().__init__(iids_configs=self.__iids_configs)
        iids_delay_label_core.ocl_pipeline(self, prediction_class=prediction_class)

    def Fog_Layer_IIDS(self,
                       base_device: str,
                       prediction_class='normal') -> None:
        assert base_device.lower() in params.edge_device_list, 'No base_device name found in edge device list'
        base_device = base_device.lower()

        self.__iids_configs.update({"base_device": base_device})
        self.__iids_configs.update({"device_in_fusion": []})
        self.__iids_configs.update({"layer": params.fog_layer})
        super().__init__(iids_configs=self.__iids_configs)
        iids_delay_label_core.ocl_pipeline(self, prediction_class=prediction_class)