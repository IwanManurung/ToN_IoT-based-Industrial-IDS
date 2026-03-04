from . import iids_util
from base import util
import pandas as pd
import numpy as np
import os
import time
from typing import List, Dict, Union, Callable
# import time
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

class iids_core:
    def __init__(self,
                 iids_configs: Dict):
        self.__load_configs(iids_configs=iids_configs)
        self.__make_dataset()
        self.__feature_normalization()
        pass
    
    @property
    def base_device(self):
        return self.__base_device
    
    @property
    def all_in_fusion(self):
        return self.__all_in_fusion
    
    @property
    def device_in_fusion(self):
        return self.__device_in_fusion
    
    @property
    def layer(self):
        return self.__layer
    
    @property
    def prediction_class(self):
        return self.__prediction_class
    
    @property
    def anomaly_models(self):
        return self.__anomaly_models
    
    @property
    def classifier_models(self):
        return self.__classifier_models
    
    @property
    def classifier_models(self):
        return self.__classifier_models
    
    @property
    def proba_threshold(self):
        return self.__proba_threshold
    
    @property
    def similarity_threshold(self):
        return self.__similarity_threshold
    
    @property
    def annot_delay(self):
        return self.__annot_delay
    
    @property
    def normalized(self):
        return self.__normalized
    
    @property
    def scaler_model(self):
        return self.__scaler_model
    
    @property
    def scaler_model(self):
        return self.__scaler_model
    
    @property
    def window_size(self):
        return self.__window_size
    
    @property
    def random_seed(self):
        return self.__random_seed
        
    @property
    def run_audited_version(self):
        return self.__run_audited_version

    @property
    def audited_size(self):
        return self.__audited_size
    
    @property
    def header(self):
        return self.__header
    
    @property
    def feature(self):
        return self.__feature
    
    @property
    def target(self):
        return self.__target
    
    @property
    def available_prediction_class(self):
        return list(set(self.target))
    
    @property
    def __precalc_feature_file(self):
        file = f"src/precalc/{self.layer}_{self.base_device}_{self.scaler_model}_{self.window_size}.npz"
        return file
    
    @property
    def __make_precalc_scaled_feature(self):
        dst = self.__precalc_feature_file
        check1 = False
        check2 = False
        if os.path.exists(dst):
            loaded = np.load(dst, allow_pickle=True)
            scaled = loaded.get('scaled')
            header = loaded.get('header')
            # consistency check
            check1 = scaled.shape == self.__feature.shape
            check2 = len(set(header) - set(self.header)) == 0
        
        if check1 & check2:
            return scaled
        else:
            scaled = iids_util.online_normalization(data=self.feature,
                                                    window_size=self.window_size,
                                                    scaler_model=self.scaler_model)
                
            np.savez(self.__precalc_feature_file,
                     scaled=scaled,
                     header=self.header)
            return scaled

    def __load_configs(self, iids_configs):
        self.__base_device = iids_configs.get("base_device")
        self.__all_in_fusion = iids_configs.get("all_in_fusion")
        self.__device_in_fusion = iids_configs.get("device_in_fusion")
        self.__layer = iids_configs.get("layer")
        self.__prediction_class = iids_configs.get("prediction_class")
        self.__anomaly_models = iids_configs.get("anomaly_models")
        self.__classifier_models = iids_configs.get("classifier_models")
        self.__proba_threshold = iids_configs.get("proba_threshold")
        self.__similarity_threshold = iids_configs.get("similarity_threshold")
        self.__annot_delay = iids_configs.get("annot_delay")
        self.__normalized = iids_configs.get("normalized")
        self.__scaler_model = iids_configs.get("scaler_model")
        self.__window_size = iids_configs.get("window_size")
        self.__random_seed = iids_configs.get("random_seed")
        self.__low_memory = iids_configs.get("low_memory")
        self.__run_audited_version = iids_configs.get("run_audited_version")
        self.__audited_size = iids_configs.get("audited_size")

    def __make_dataset(self):
        # Step 1: Loading and formatting the relevant dataset
        util.nice_print(msg=f"Step 1: Preparing dataset", align='left')
        if self.layer == 'edge':
            feature, target, header = iids_util.loading_edge_dataset(
                base_device=self.base_device,
                all_in_fusion=self.all_in_fusion,
                others=self.device_in_fusion,
                load_all=not self.run_audited_version,
                sample_size=self.audited_size,
                random_seed=self.random_seed)
        elif self.layer == 'fog':
            feature, target, header = iids_util.loading_fog_dataset(
                base_device=self.base_device,
                load_all=not self.run_audited_version,
                sample_size=self.audited_size,
                random_seed=self.random_seed)
        if self.__low_memory:
            feat_df = pd.DataFrame(feature, columns=header)
            for col in feat_df:
                absmax = np.abs(feat_df[col].tolist()).max()
                if absmax > 1_000:
                    feat_df[col] = np.divide(feat_df[col].tolist(), 1_000)
                feat_df[col] = feat_df[col].astype(np.float16)
            self.__feature = feat_df.to_numpy(dtype=np.float16).copy()
            del feat_df
            self.__target = target
            self.__header = header
        else:
            self.__feature = feature
            self.__target = target
            self.__header = header
    
    @property
    def label_distribution(self):
        return self.__label_dist
    
    @property
    def encoded_target(self):
        return self.__encoded_target
    
    def __feature_normalization(self):
        if self.normalized:
            # Check if precalculated normalized exists, if no, make one
            self.__feature = self.__make_precalc_scaled_feature       
        else:
            # if Normalized is False, No changed to feature
            pass
            util.nice_print(msg=f'. Proceeds Using Raw Feature', align='left')

    def __make_stream(self):
        from capymoa.stream import NumpyStream
        encoded_target = iids_util.map_as_binary_class(ntarget=self.__target,
                                                       class_0=self.prediction_class)
        self.__encoded_target = encoded_target

        class0 = round(self.__target.count(self.prediction_class) / len(self.__target), 2)
        class1 = 1 - class0

        self.__label_dist = {self.prediction_class: class0,
                             f"NOT-{self.prediction_class}": class1}
        
        self.Istream = NumpyStream(X=self.__feature,
                                   y=encoded_target)
        self.Dstream = NumpyStream(X=self.__feature,
                                   y=encoded_target)

    def __make_learning_pipeline(self):
        # Setup 6: Setup Anomaly and Classifier models
        detector_1, detector_2, detector_3 = iids_util.setup_learning_models(iids_model="AnomalyModel",
                                                                             stream=self.Istream,
                                                                             models=self.anomaly_models,
                                                                             random_seed=self.random_seed)
        # InTimeLearner: Ipredict and Itrain represent Online Anomaly Detector
        self.__Iproba = lambda ins: (detector_1.score_instance(ins), detector_2.score_instance(ins), detector_3.score_instance(ins))
        self.__Itrain = lambda ins: (detector_1.train(ins), detector_2.train(ins), detector_3.train(ins))
        util.print_list(nlist=self.anomaly_models, startswith='.. Loaded Anomaly models:')

        classifier_1, classifier_2, classifier_3 = iids_util.setup_learning_models(iids_model="ClassifierModel",
                                                                             stream=self.Dstream,
                                                                             models=self.classifier_models,
                                                                             random_seed=self.random_seed)
        # DelayLearner: Dpredict and Dtrain represent Delayed Classifier
        self.__Dpredict = lambda ins: (classifier_1.predict(ins), classifier_2.predict(ins), classifier_3.predict(ins))
        self.__Dtrain = lambda ins: (classifier_1.train(ins), classifier_2.train(ins), classifier_3.train(ins))
        util.print_list(nlist=self.classifier_models, startswith='.. Loaded Classifier models:')

    def run_ocl(self,
                scenario):
        self.__make_stream()
        self.__make_learning_pipeline()
        if scenario == 1:
            self.__ocl_anomaly()
        elif scenario == 2:
            self.__ocl_supervised()
        elif scenario == 3:
            self.__ocl_semi_supervised()

    @property
    def metric_dicts(self):
        return self.__metric_dicts
    
    def __evaluator(self, model_output, runtime):
        self.__reset_evaluator()

        if len(model_output[0]) == 5:
            model_id = ['m1', 'm2', 'm3'] 
        elif len(model_output[0]) == 8:
            model_id = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6']

        columns = model_id + ['y_predict', 'y_true']

        metric_dicts = {"overall": {},
                       "by-model": {},
                       "agreement": {}}
        
        table = pd.DataFrame(model_output, columns=columns)
        y_true = table['y_true'].tolist().copy()
        y_pred = table['y_predict'].tolist().copy()
        overall = iids_util.evaluation_metrics(y_true=y_true, y_pred=y_pred)
        overall.update({"runtime": runtime})
        overall.update({"Instances_num": len(model_output)})
        # add overall metrics
        metric_dicts.update({"overall": overall})
        # add metrics per models
        for col in model_id:
            y_pred = table[col].tolist().copy()
            metrics = iids_util.evaluation_metrics(y_true=y_true, y_pred=y_pred)
            metric_dicts.get("by-model").update({col: metrics})
            del metrics, y_pred
        # add agreement score between models
        for x1,x2 in combinations(columns, r=2):
            y1 = table[x1].tolist().copy()
            y2 = table[x2].tolist().copy()
            score = cohen_kappa_score(y1=y1, y2=y2)
            metric_dicts.get('agreement').update({f"{x1}-{x2}": round(score,3)})
        self.__metric_dicts = metric_dicts

    def __reset_majority_class(self):
        self.__class_counter = []

    def __update_majority_class(self, y=None):
        if y == None:
            pass
        else:
            self.__class_counter.append(y)
            if len(self.__class_counter) > self.window_size:
                self.__class_counter = self.__class_counter[1:]
        
        return int(self.__class_counter.count(1) > self.__class_counter.count(0))

    def __reset_evaluator(self):
        self.__metric_dicts = {"overall": {},
                               "by-model": {},
                               "agreement": {}}
        
    def __ocl_anomaly(self):
        print("run OCL Anomaly")
        instances_seen = 0
        err_instances = 0
        model_output = []
        self.Istream.restart()
        self.__reset_majority_class()
        start = time.time() # runtime start at the beginning of the test-then-train loops
        while self.Istream.has_more_instances():
            try:
                curr_instance = self.Istream.next_instance()
                # first, model to test the current instance
                proba_score = list(self.__Iproba(curr_instance))
                # replace NA with majority class if any
                majority_class = self.__update_majority_class()
                proba_score = iids_util.proba_rules(proba_score)
                # proba_score = [majority_class if item == None else item for item in proba_score]
                y_models = iids_util.proba_prediction_rules(nscore=proba_score,
                                                            threshold=self.proba_threshold)
                
                y_predict = iids_util.voting_decision(npredicts=y_models)
                # update majority class
                self.__update_majority_class(y=y_predict)
                # y_models = output models, final decision, y_true
                output = y_models + [y_predict] + [curr_instance.y_index]
                model_output.append(output.copy())
                # then, model to train the current instance
                self.__Itrain(curr_instance)
                instances_seen += 1
                # delete temporary variable to ensure no repeated value
                del y_models, y_predict, output
            except:
                err_instances += 1
                # print(curr_instance.x, end='\r' ,flush=True)

            if (instances_seen % 1000 == 0) or (self.Istream.has_more_instances() == False):
                progress = instances_seen/self.Istream._len
                msg1 = util.progress_meter(progress=progress)
                msg2 = f' ~ Instances seen: {instances_seen:,}. Error: {err_instances:,}'
                print(f'>> {msg1} {msg2}', end='\r', flush=True)
            runtime = round(time.time() - start, 2)
        self.__evaluator(model_output=model_output,
                         runtime=runtime)

    def __ocl_supervised(self):
        print("run OCL Supervised")
        instances_seen = 0
        err_instances = 0
        model_output = []
        self.Dstream.restart()
        self.__reset_majority_class()
        start = time.time() # runtime start at the beginning of the test-then-train loops
        while self.Istream.has_more_instances():
            try:
                curr_instance = self.Istream.next_instance()
                # first, model to test the current instance
                dpredicts = list(self.__Dpredict(curr_instance))
                # replace NA with zero if any
                majority_class = self.__update_majority_class()
                dpredicts = [majority_class if item == None else item for item in dpredicts]
                y_models = iids_util.proba_prediction_rules(nscore=dpredicts,
                                                            threshold=self.proba_threshold)
                
                y_predict = iids_util.voting_decision(npredicts=y_models)
                
                # update majority class
                self.__update_majority_class(y=y_predict)
                # y_models = output models, final decision, y_true
                output = y_models + [y_predict] + [curr_instance.y_index]
                model_output.append(output.copy())
                # then, model to train the current instance
                self.__Dtrain(curr_instance)
                instances_seen += 1
                # delete temporary variable to ensure no repeated value
                del y_models, y_predict, output
            except:
                err_instances += 1
                # print(curr_instance.x, end='\r' ,flush=True)

            if (instances_seen % 1000 == 0) or (self.Istream.has_more_instances() == False):
                progress = instances_seen/self.Istream._len
                msg1 = util.progress_meter(progress=progress)
                msg2 = f' ~ Instances seen: {instances_seen:,}. Error: {err_instances:,}'
                print(f'>> {msg1} {msg2}', end='\r', flush=True)
            runtime = round(time.time() - start, 2)
        self.__evaluator(model_output=model_output,
                         runtime=runtime)
        pass

    def __ocl_semi_supervised(self):
        print("run OCL Semi Supervised")
        instances_seen = 0
        err_instances = 0
        label_seen = 0
        model_output = []
        self.Dstream.restart()
        self.Istream.restart()
        self.__reset_majority_class()
        start = time.time() # runtime start at the beginning of the test-then-train loops
        while self.Istream.has_more_instances():
            try:
                curr_instance = self.Istream.next_instance()
                majority_class = self.__update_majority_class()
                # first, predict with InTimeLearner
                proba_score = list(self.__Iproba(curr_instance))
                # train with InTimeLearner
                self.__Itrain(curr_instance)
                instances_seen += 1
                # check if delayed label exists.
                # if delayed label not exist, run IIDS only with InTimeLearner
                proba_score = iids_util.proba_rules(proba_score)
                # proba_score = [max(proba_score) if item == None else item for item in proba_score]
                y_Imodels = iids_util.proba_prediction_rules(nscore=proba_score,
                                                            threshold=self.proba_threshold)
                if instances_seen <= self.annot_delay:
                    # Since DelayLearner is inactive,
                    # assumed its output is identical with InTimeLearner
                    y_Dmodels = y_Imodels
                else:
                    # DelayLearner adopts train-then-test techniques
                    # First, proceed train
                    D_instance = self.Dstream.next_instance()
                    self.__Dtrain(D_instance)
                    label_seen += 1
                    # proceed predict
                    y_Dmodels = self.__Dpredict(curr_instance)
                    y_Dmodels = [majority_class if item == None else item for item in y_Dmodels]
                    # Find similarity between current InTime instance and delay instance
                    stream_similarity = cosine_similarity([curr_instance.x], [D_instance.x])[0][0]
                    if stream_similarity > self.similarity_threshold:
                        # if both stream highly similar,
                        y_Imodels = y_Dmodels # discard prediction by InTimeLearner

                # final prediction is only by InTimeLearner
                y_models = y_Imodels + y_Dmodels
                y_predict = iids_util.voting_decision(npredicts=y_models)
                # Update majority class
                self.__update_majority_class(y=y_predict)
                # y_models = output models, final decision, y_true
                output = y_models + [y_predict] + [curr_instance.y_index]
                model_output.append(output.copy())
                # delete temporary variable to ensure no repeated value
                del y_models, y_predict, output
            except:
                err_instances += 1

            if (instances_seen % 1000 == 0) or (self.Istream.has_more_instances() == False):
                progress = instances_seen/self.Istream._len
                msg1 = util.progress_meter(progress=progress)
                msg2 = f' ~ Instances seen: {instances_seen:,}. Label seen: {label_seen:,}. Error: {err_instances:,}'
                print(f'>> {msg1} {msg2}', end='\r', flush=True)
            runtime = round(time.time() - start, 2)
        self.__evaluator(model_output=model_output,
                         runtime=runtime)