from ._iids_core import iids_core
from base import params, util
import json
from typing import List, Literal, Dict, Union
# 
class ToN_IoT_IIDS(iids_core):

    def __init__(self,
                 layer,
                 base_device,
                 all_in_fusion =True,
                 device_in_fusion =[],
                 anomaly_models =['HalfSpaceTrees', 'Autoencoder', 'AdaptiveIsolationForest'],
                 classifier_models =['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree'],
                 proba_threshold =.6,
                 normalized =True,
                 scaler_model ='StandardScaler',
                 window_size =60,
                 random_seed =1,
                 low_memory =True,
                 run_audited_version =True,
                 audited_size =.1) -> None:
        
        iids_configs = {}
        iids_configs.update({"layer": layer})
        iids_configs.update({"base_device": base_device})
        iids_configs.update({"all_in_fusion": all_in_fusion})
        iids_configs.update({"device_in_fusion": device_in_fusion})
        iids_configs.update({"anomaly_models": anomaly_models})
        iids_configs.update({"classifier_models": classifier_models})
        iids_configs.update({"proba_threshold": proba_threshold})
        iids_configs.update({"normalized": normalized})
        iids_configs.update({"scaler_model": scaler_model})
        iids_configs.update({"window_size": window_size})
        iids_configs.update({"random_seed": random_seed})
        iids_configs.update({"run_audited_version": run_audited_version})
        iids_configs.update({"audited_size": audited_size})
        self.__iids_configs = iids_configs
        super().__init__(iids_configs=self.__iids_configs)

    def run_Anomaly_Detector(self,
                             prediction_class,
                             anomaly_models =['HalfSpaceTrees', 'Autoencoder', 'AdaptiveIsolationForest'],
                             proba_threshold =.5,
                             ):
        
        self.__iids_configs.update({"prediction_class": prediction_class})
        self.__iids_configs.update({"anomaly_models": anomaly_models})
        self.__iids_configs.update({"proba_threshold": proba_threshold})
        super().__init__(iids_configs=self.__iids_configs)
        iids_core.run_ocl(self, scenario=1)

    def run_Supervised(self,
                       prediction_class,
                       classifier_models =['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree']):
        
        self.__iids_configs.update({"prediction_class": prediction_class})
        self.__iids_configs.update({"classifier_models": classifier_models})
        super().__init__(iids_configs=self.__iids_configs)
        iids_core.run_ocl(self, scenario=2)

    def run_semi_Supervised_Delay_Label(self,
                                        prediction_class,
                                        anomaly_models =['HalfSpaceTrees', 'Autoencoder', 'AdaptiveIsolationForest'],
                                        classifier_models =['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree'],
                                        annot_delay =300,
                                        proba_threshold =.5,
                                        similarity_threshold=.5,
                                        ):

        self.__iids_configs.update({"prediction_class": prediction_class})
        self.__iids_configs.update({"anomaly_models": anomaly_models})
        self.__iids_configs.update({"proba_threshold": proba_threshold})
        self.__iids_configs.update({"classifier_models": classifier_models})
        self.__iids_configs.update({"similarity_threshold": similarity_threshold})
        self.__iids_configs.update({"annot_delay": annot_delay})
        super().__init__(iids_configs=self.__iids_configs)
        iids_core.run_ocl(self, scenario=3)
        