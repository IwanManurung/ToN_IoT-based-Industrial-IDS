from capymoa.stream import NumpyStream
from . import iids_util
from base import util
import pandas as pd
import numpy as np
from typing import List, Dict, Union, Callable
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
'''
ONE vs RestOfTheClass
'''
class iids_delay_label_core:
    def __init__(self, iids_configs):
        self.__load_configs(iids_configs=iids_configs)
        util.nice_print(msg=f"ToN_IoT based Industrial ICS", align='center')
        util.nice_print(msg=f"Scenario: Classification with Delayed Label", align='center')
        util.nice_print(msg=f"Random seed: {self.random_seed}", align='center')
        util.print_list(nlist=self.anomaly_models, startswith='Anomaly models:', style="default")
        util.print_list(nlist=self.classifier_models, startswith='Classifier models:', style="default")
        self.__make_dataset()
        pass
    
    def evaluate_majority_class(self, 
                                y_index: int =None):
        if type(y_index) != type(None):
            self.__rolling_yindex.append(y_index)
            if len(self.__rolling_yindex) > self.window_size:
                self.__rolling_yindex = self.__rolling_yindex[1:]
        
        if self.__rolling_yindex.count(1) > self.__rolling_yindex.count(0):
            return 1
        else:
            return 0
        
    def __is_precalc_feature_exist(self):
        dst = f"src/precalc/{self.layer}_{self.base_device}_{self.scaler_model}_{self.window_size}.npz"
        return os.path.exists(dst)

    def __load_from_precalc_scaled_feature(self):
        dst = f"src/precalc/{self.layer}_{self.base_device}_{self.scaler_model}_{self.window_size}.npz"
        loaded = np.load(dst, allow_pickle=True)
        scaled = loaded.get('scaled')
        header = loaded.get('header')
        # consistency check
        if (len(scaled) == len(self.__feature)) & (len(set(header) - set(self.header)) == 0):
            return scaled

    def __load_configs(self, iids_configs):
        self.base_device = iids_configs.get("base_device")
        self.all_in_fusion = iids_configs.get("all_in_fusion")
        self.device_in_fusion = iids_configs.get("device_in_fusion")
        self.layer = iids_configs.get("layer")
        self.anomaly_models = iids_configs.get("anomaly_models")
        self.classifier_models = iids_configs.get("classifier_models")
        self.delayed_label = iids_configs.get("delayed_label")
        self.normalized = iids_configs.get("normalized")
        self.low_memory = iids_configs.get("low_memory")
        self.scaler_model = iids_configs.get("scaler_model")
        self.window_size = iids_configs.get("window_size")
        self.random_seed = iids_configs.get("random_seed")
        self.run_audited_version = iids_configs.get("run_audited_version")
        self.audited_size = iids_configs.get("audited_size")
        # print("Load Config: Done")
    
    def __make_dataset(self):
        # Step 1: Loading and formatting the relevant dataset
        util.nice_print(msg=f"Step 1: Preparing dataset", align='left')
        if self.layer == 'edge':
            feature, target, header = iids_util.loading_edge_dataset(
                base_device=self.base_device,
                all_in_fusion=self.all_in_fusion,
                others=self.device_in_fusion,
                load_all=not self.run_audited_version,
                low_memory=self.low_memory,
                sample_size=self.audited_size,
                random_seed=self.random_seed)
        elif self.layer == 'fog':
            feature, target, header = iids_util.loading_fog_dataset(
                base_device=self.base_device,
                load_all=not self.run_audited_version,
                low_memory=self.low_memory,
                sample_size=self.audited_size,
                random_seed=self.random_seed)
            
        self.__feature = feature
        self.__target = target
        self.header = header
        self.available_prediction_class = list(set(target))

        # Print running text info
        dev = self.base_device.upper()
        if self.device_in_fusion == []:
            fusion = 'with No Fusion data'
        else:
            fusion = 'with Fusion data in ' + ','.join([i.upper() for i in self.device_in_fusion])
        msg = f"  Edge Device: {dev} {fusion} with {dev} Label"
        util.nice_print(msg, align='left')
        msg = f"  Loaded dataset: {len(self.__target):,} with {len(self.header)} features"
        util.nice_print(msg, align='left')
        util.print_list(nlist=self.header, startswith=f'. Feature names:', style='upper')
        util.print_list(nlist=self.available_prediction_class, startswith=f'. Available prediction classes:', style='upper')

    def __make_stream_pipeline(self, prediction_class):
        # Step 2: Mapping the prediction target
        util.nice_print(msg=f"Step 2: Preparing Feature and Target Encoding", align='left')
        self.encoded_target = iids_util.map_as_binary_class(ntarget=self.__target,
                                                            class_0=prediction_class)

        # Encoding the target label into numeric representation
        # class_0 always 0 and rest of the class always 1
        self.encoding_rules = {prediction_class: 0,
                               f"NOT_{prediction_class}": 1}
        
        len_target = len(self.encoded_target)
        class_0 = f"{prediction_class.upper()} [0]"
        class_1 = f"NOT_{prediction_class.upper()} [1]"
        class_dist = {class_0: round(100 * self.encoded_target.count(0) / len_target,2),
                      class_1: round(100 * self.encoded_target.count(1) / len_target, 2)}
        util.nice_print(msg=f". Prediction class: {prediction_class.upper()}", align='left')
        util.print_dict(dicts=class_dist,
                        startswith=". Target Class Distribution:",
                        sort_value=False)
        
        # Step 4: Data Windowing, only required if Normalized is True
        util.nice_print(msg=f"Step 4: Setup Normalization", align='left')
        if self.normalized:
            # Do online normalization to feature, bin by bin
            if self.__is_precalc_feature_exist():
                self.feature = self.__load_from_precalc_scaled_feature()
                print("load from precalc")
            else:
                self.feature = iids_util.online_normalization(data=self.__feature,
                                                              window_size=self.window_size,
                                                              scaler_model=self.scaler_model)
                
                dst = f"src/precalc/{self.layer}_{self.base_device}_{self.scaler_model}_{self.window_size}.npz"
                np.savez(dst, scaled=self.feature, header=self.header)
                print("Calc and saved scaled feature")
                
            util.nice_print(msg=f'. Proceeds Feature Normalization using sklearn {self.scaler_model} with Window size: {self.window_size}', align='left')
        else:
            # if Normalized is False, No changed to feature
            self.feature = self.__feature.copy()
            util.nice_print(msg=f'. Proceeds Using Raw Feature', align='left')
        
        # Step 5: Preparing dataset as online streaming data
        util.nice_print(msg=f"Step 5: Make Online Streaming datasets", align='left')
        self.Istream = NumpyStream(X=self.feature,
                                  y=self.encoded_target)
        self.Dstream = NumpyStream(X=self.feature,
                                  y=self.encoded_target)
        
    def __make_learning_pipeline(self):
        # Setup 6: Setup Anomaly and Classifier models
        util.nice_print(msg=f"Step 6: Setting up Hybrid Detection models", align='left')
        util.nice_print(msg=f".. Setting up Hybrid Detection models", align='left')

        detector_1, detector_2, detector_3 = iids_util.setup_learning_models(iids_model="AnomalyModel",
                                                                             stream=self.Istream,
                                                                             models=self.anomaly_models,
                                                                             random_seed=self.random_seed)
        # InTimeLearner: Ipredict and Itrain represent Online Anomaly Detector
        self.__Iproba = lambda ins: (detector_1.score_instance(ins), detector_2.score_instance(ins), detector_3.score_instance(ins))
        self.__Itrain = lambda ins: (detector_1.train(ins), detector_2.train(ins), detector_3.train(ins))
        util.print_list(nlist=self.anomaly_models, startswith='.. Loaded Anomaly models:')

        classifier_1, classifier_2, classifier_3 = iids_util.setup_learning_models(iids_model="ClassifierModel",
                                                                             stream=self.Istream,
                                                                             models=self.classifier_models,
                                                                             random_seed=self.random_seed)
        # DelayLearner: Dpredict and Dtrain represent Delayed Classifier
        self.__Dpredict = lambda ins: (classifier_1.predict(ins), classifier_2.predict(ins), classifier_3.predict(ins))
        self.__Dtrain = lambda ins: (classifier_1.train(ins), classifier_2.train(ins), classifier_3.train(ins))
        util.print_list(nlist=self.classifier_models, startswith='.. Loaded Classifier models:')
    
    def stream_restart(self):
        self.Istream.restart()
        self.Dstream.restart()
    
    def ocl_pipeline(self, prediction_class):
        self.__make_stream_pipeline(prediction_class=prediction_class)
        self.__make_learning_pipeline()
        util.nice_print(msg=f"Step 7: Proceeding test-then-train based OCL ...", align='left')
        Istream_seen = 0
        Dstream_seen = 0
        model_output = []
        self.stream_restart()
        self.__rolling_yindex = []
        start = time.time() # runtime start at the beginning of the test-then-train loops
        while self.Istream.has_more_instances():
            curr_instance = self.Istream.next_instance()
            # If delayed label did not exist, run IIDS with anomaly models only
            try:
                if Istream_seen < self.delayed_label:
                    # first, InTimeLearner to run the test-then-train
                    y_Iproba = list(self.__Iproba(curr_instance))
                    y_Imodels = iids_util.proba_prediction_rules(nscore=y_Iproba)
                    # Since DelayLearner is inactive, y_models was defined only by InTimeLearner
                    y_Dmodels = y_Imodels
                    # then, model to train the current instance
                    self.__Itrain(curr_instance)
                    # Decision on anomaly models
                    yI_predict = iids_util.voting_decision(y_Imodels,
                                                    rep_NA=False)
                    # update rolling majority class with yO_predict
                    self.evaluate_majority_class(y_index=yI_predict)
                else:
                    # DelayLearner is active. DelayLearner able to use train-then-test procedure
                    # First step, run DelayLearner to train DelayStream
                    # then calculate similarity between current Istream and current Dstream
                    D_instance = self.Dstream.next_instance()
                    self.__Dtrain(D_instance)
                    # update rolling majority class with delayed label
                    self.evaluate_majority_class(y_index=D_instance.y_index)
                    # print('curr',curr_instance.x)
                    # print('D',D_instance.x)
                    stream_similarity = cosine_similarity([curr_instance.x], [D_instance.x])[0][0]
                    if stream_similarity > .99:
                        # if both stream highly similar, then let DelayLearner make the predictions
                        y_Dmodels = list(self.__Dpredict(curr_instance))
                        y_Imodels = y_Dmodels
                    else:
                        # else, let both learner make the predictions
                        y_Iproba = list(self.__Iproba(curr_instance))
                        self.__Itrain(curr_instance)
                        y_Imodels = iids_util.proba_prediction_rules(nscore=y_Iproba)
                        y_Dmodels = list(self.__Dpredict(curr_instance))
                    Dstream_seen += 1
                Istream_seen += 1
            except:
                # print(curr_instance)
                pass
            
            # get current rolling majority class
            majority_class = self.evaluate_majority_class()
            # y_models were defined by both learner
            y_models = y_Imodels + y_Dmodels
            y_final = iids_util.voting_decision(y_models,
                                                rep_NA=True,
                                                rep_NA_as=majority_class)
            
            # Output table 
            output = y_models + [y_final] + [curr_instance.y_index]
            model_output.append(output.copy())
            # delete temporary variable to ensure no repeated value
            # del y_Imodels, y_Dmodels, output, y_models, y_final

            # print progress
            if (Istream_seen % 1000 == 0) or (self.Istream.has_more_instances() == False):
                progress = Istream_seen/self.Istream._len
                msg = util.progress_meter(progress=progress) + f' ~ Instances seen: {Istream_seen:,}. Label seen: {Dstream_seen:,}'
                print(f'>> {msg}', end='\r', flush=True)
        # yO_models = output models, final decision, y_true
        runtime = round(time.time() - start, 2)
        util.nice_print(msg=f". Reviewing {Istream_seen:,} were done", align='left')
        self.evaluator(model_output=model_output,
                       runtime=runtime)

    def evaluator(self, model_output, runtime):
        columns = ['detector1', 'detector2', 'detector3',
                   'classifier1', 'classifier2', 'classifier3',
                   'y_predict', 'y_true']
        table = pd.DataFrame(model_output, columns=columns)

        metrics = iids_util.evaluation_metrics(y_true=table['y_true'].tolist(),
                                               y_pred=table['y_predict'].tolist())
        metrics.update({"runtime": runtime})
        metrics.update({"Instances_num": len(model_output)})

        self.metrics = metrics
        self.model_output = table
        util.print_dict(dicts=metrics, startswith="Evaluation Metrics: ")