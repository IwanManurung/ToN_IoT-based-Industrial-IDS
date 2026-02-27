from . import iids_supervised_core
from base import params, util
import json
from typing import List, Literal, Dict, Union
# 
class ICS_IIDS_Supervised(iids_supervised_core):

    def __init__(self,
                 classifier_models: List[str] =['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree'],
                 normalized: bool =True,
                #  low_memory: bool =False,
                 scaler_model: Literal['StandardScaler', 'MinMaxScaler']='StandardScaler',
                 window_size: int =600,
                 random_seed: int =1,
                 run_audited_version: bool =True,
                 audited_size: float =0.2) -> None:

        assert len(set(classifier_models)) == 3, "Learning models should be 3 unique models"
        assert window_size > 0
        assert random_seed > 0
        
        configs = {}
        configs.update({"classifier_models": classifier_models})
        configs.update({"normalized": normalized})
        # configs.update({"low_memory": low_memory})
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
                        all_in_fusion: bool =True,
                        device_in_fusion: List[str] =[],
                        prediction_class='normal') -> None:
        assert base_device.lower() in params.edge_device_list, 'No base_device name found in edge device list'
        base_device = base_device.lower()
        self.__iids_configs.update({"base_device": base_device})
        self.__iids_configs.update({"all_in_fusion": all_in_fusion})
        self.__iids_configs.update({"device_in_fusion": device_in_fusion})
        self.__iids_configs.update({"layer": params.edge_layer})
        super().__init__(iids_configs=self.__iids_configs)
        iids_supervised_core.ocl_pipeline(self, prediction_class=prediction_class)

    def Fog_Layer_IIDS(self,
                       base_device: str,
                       prediction_class='normal') -> None:
        assert base_device.lower() in params.edge_device_list, 'No base_device name found in edge device list'
        base_device = base_device.lower()

        self.__iids_configs.update({"base_device": base_device})
        self.__iids_configs.update({"device_in_fusion": []})
        self.__iids_configs.update({"layer": params.fog_layer})
        super().__init__(iids_configs=self.__iids_configs)
        iids_supervised_core.ocl_pipeline(self, prediction_class=prediction_class)