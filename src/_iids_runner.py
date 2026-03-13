from ._iids_core import iids_core
from base import params, util
import json
from typing import List, Literal, Dict, Union
from sklearn.metrics import cohen_kappa_score
# 
class ToN_IoT_IIDS(iids_core):

    def __init__(self,
                 base_device,
                 scenario: Literal[1,2,3],
                 all_in_fusion =True,
                 device_in_fusion =[],
                 anomaly_models =['HalfSpaceTrees', 'Autoencoder', 'AdaptiveIsolationForest'],
                 classifier_models =['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree'],
                 proba_threshold =.75,
                 normalized =True,
                 scaler_model ='StandardScaler',
                 window_size =60,
                 random_seed =1,
                 annot_delay =300,
                 similarity_threshold =.5,
                 run_audited_version =True,
                 audited_size =.1) -> None:
        
        iids_configs = {}
        iids_configs.update({"scenario": scenario})
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
        iids_configs.update({"annot_delay": annot_delay})
        iids_configs.update({"similarity_threshold": similarity_threshold})
        iids_configs.update({"run_audited_version": run_audited_version})
        iids_configs.update({"audited_size": audited_size})
        self.__iids_configs = iids_configs
    
    @property
    def edge_fog_agreement(self):
        return self.__agreement
    
    def run_test_then_train(self, prediction_class):
        util.nice_print(msg="", align='center')
        util.nice_print(msg="", align='center')
        util.nice_print(msg="ToN_IoT-based Industrial-IDS", align='center')
        util.nice_print(msg="On-device Threat Detection at Edge Layer", align='center')
        self.__iids_configs.update({"prediction_class": prediction_class})

        # running Edge-IIDS
        self.__iids_configs.update({"layer": 'edge'})
        scenario = self.__iids_configs.get('scenario')
        util.nice_print(msg=f"Running Edge-IIDS Scenario-{scenario}", align='left')
        util.print_dict(self.__iids_configs, startswith='Industrial-IDS Parameters', sort_value=False)
        super().__init__(iids_configs=self.__iids_configs)
        iids_core.run_ocl(self, scenario=scenario)
        self.edge_eval = iids_core.metrics_eval(self)
        edge_predict = iids_core.y_output(self)

        # running Fog-IIDS
        self.__iids_configs.update({"layer": 'fog'})
        scenario = self.__iids_configs.get('scenario')
        util.nice_print(msg=f"Running Fog-IIDS Scenario-{scenario}", align='left')
        util.print_dict(self.__iids_configs, startswith='Industrial-IDS Parameters', sort_value=False)
        super().__init__(iids_configs=self.__iids_configs)
        iids_core.run_ocl(self, scenario=scenario)
        self.fog_eval = iids_core.metrics_eval(self)
        fog_predict = iids_core.y_output(self)

        self.__agreement= cohen_kappa_score(y1=edge_predict, y2=fog_predict)