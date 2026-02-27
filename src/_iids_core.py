from capymoa.stream import NumpyStream
from . import iids_base
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from base import util
import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, LabelEncoder, MinMaxScaler)
from itertools import pairwise
import time
# import json
from base import params
from datastream import fusion_engine
from typing import List, Dict, Union, Callable

class iids_core:
    def __init__(self,
                 iids_configs: Dict[str, Union[List[str], str, float, int]]) -> None:
        self.__load_iids_config(iids_configs=iids_configs)
        self.__waiting_time = 0
        self.__make_stream_pipeline()
        self.__make_learner_pipeline()
        self.__ocl_pipeline()
    
    @staticmethod
    def __encoding_target_label(ntarget: List[str]) -> Union[List[int], Dict[str, int]]:
        # encoding target and save the encoding rules
        le = LabelEncoder()
        encoded_target = le.fit_transform(ntarget)
        encoding_rules = {}
        [encoding_rules.update({i: int(le.transform([i])[0])}) for i in set(ntarget)]
        return encoded_target, encoding_rules

    @staticmethod
    def __target_distribution(nlist: List) -> Dict[str, float]:
        dist_percent = {}
        for _ in set(nlist):
            rasio = round(100 * nlist.count(_) / len(nlist), 3)
            dist_percent.update({_: rasio})
        return dist_percent
    
    @staticmethod
    def __load_scaler_model(model: str ='StandardScaler') -> Callable:
        if model.lower() == 'MinMaxScaler'.lower():
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        return scaler
    
    @staticmethod
    def __data_windowing(nlist: List,
                         window_size: int) -> List:
        bin_range = list(range(0, len(nlist), int(window_size)))
        bin_range.append(len(nlist))
        bin = list(pairwise(bin_range))
        bins = [nlist[start:end] for start,end in bin]
        return bins
    
    @staticmethod
    def __binary_evaluator(y_true: List,
                           y_pred: List) -> Dict[str, float]:
        assert len(y_true) == len(y_pred), 'length of y_true must same as y_pred'
        assert sorted(set(y_true)) == sorted(set(y_true)), 'Unique items in y_true must same as y_pred'
        # list of evaluation metrics
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        kappa = cohen_kappa_score(y1=y_true,y2=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred)

        rounding = lambda x : round(100 * x, 2)

        binary_evaluator = {'accuracy': rounding(acc),
                            'kappa': rounding(kappa),
                            'f1_score': rounding(f1),
                            'matthews_corrcoef': rounding(mcc),
                            'precision_score': rounding(precision),
                            'recall_score': rounding(recall)
                            }
        return binary_evaluator

    @staticmethod
    def __multiclass_evaluator(y_true: List,
                               y_pred: List,
                               encoding_rules: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        assert len(y_true) == len(y_pred), 'length of y_true must same as y_pred'
        # list of metrics inherit from Binary Evaluator
        unique_true = list(set(y_true))
        # map 0 if 
        per_state_eval = {}
        attack_map = lambda attack_state, output: 1 if attack_state == output else 0
        for state in unique_true:
            state_true = [attack_map(state, i) for i in y_true]
            state_predict = [attack_map(state, i) for i in y_pred]
            state_eval = iids_core.__binary_evaluator(y_true=state_true, y_pred=state_predict)
            att_type = [key for key,val in encoding_rules.items() if val == state][0]
            per_state_eval.update({f'{state} {att_type.upper()}': state_eval})
            del state_true, state_predict, state_eval
        return per_state_eval
    
    @staticmethod
    def __voting_rules(npredicts: List,
                       encoded_normal: int,
                       random_seed: int) -> int:
        # npredicts = np.array(list(npredicts))
        # replace None with params.rep_None
        npredicts = np.array(list(map(lambda x: x if x != None else params.rep_None, npredicts)))
        npredicts.sort()
        # first, solve NA in prediction
        # if all None, then all prediction is assumed to be Normal
        while params.rep_None in npredicts:
            # if all predictor predict None, assumed as normal
            if npredicts[2] == params.rep_None:
                npredicts[npredicts == params.rep_None] = encoded_normal
            # if predictor2 predict None, assumed as predictor3
            elif npredicts[1] == params.rep_None:
                npredicts.put(1, npredicts[2])
            # if predictor1 predict None, it will choose randomly from predictor2 and predictor3
            elif npredicts[0] == params.rep_None:
                rand_gen = np.random.default_rng(random_seed)
                choosen = rand_gen.choice(list(npredicts[1:]), size=1)
                npredicts.put(0, choosen)
                del choosen 
            npredicts.sort()
        if len(set(npredicts)) == len(npredicts):
            rand_gen = np.random.default_rng(random_seed)
            y_final = rand_gen.choice(npredicts, size=1)
        else:
            npredicts = list(npredicts)
            y_final = [x for x in set(npredicts) if npredicts.count(x) >= 2]
        return y_final[0]

    def __load_iids_config(self,
                           iids_configs: Dict[str, Union[List, int, float, str]]):
        self.base_device = iids_configs.get('base_device')
        self.device_in_fusion = iids_configs.get('device_in_fusion')
        self.layer = iids_configs.get('layer')
        self.iids_model = iids_configs.get('iids_model')
        if self.iids_model == 'AnomalyModel':
            self.learning_models = iids_configs.get('anomaly_models')
        elif self.iids_model == 'ClassifierModel':
            self.learning_models = iids_configs.get('classifier_models')
        self.detection_class = iids_configs.get('detection_class')
        self.normalized = iids_configs.get('normalized')
        self.scaler_model = iids_configs.get('scaler_model')
        self.window_size = iids_configs.get('window_size')
        self.random_seed = iids_configs.get('random_seed')
        self.run_audited_version = iids_configs.get('run_audited_version')
        self.sample_size = iids_configs.get('sample_size')

    def __evaluator(self,
                    model_output: List[List[int]],
                    encoded_normal: int) -> None:
        # set 0 for normal and 1 for others
        colname = ['ym1','ym2','ym3','y_predict', 'y_index']
        table = pd.DataFrame(np.array(model_output), columns=colname)
        table = table.fillna(params.rep_None)
        table['count'] = 1
        dist_truth = table.groupby(colname).count().reset_index()
        dist_truth.to_csv(params.saved_truth_table, sep=';', index=False)

        # Binary Evaluation
        y_index = [iids_base.binary_map_eval(y=i, normal_state=encoded_normal) for i in table['y_index'].tolist()]
        y_predict = [iids_base.binary_map_eval(y=i, normal_state=encoded_normal) for i in table['y_predict'].tolist()]
        binary_evaluator = iids_core.__binary_evaluator(y_true=y_index, y_pred=y_predict)
        util.print_dict(binary_evaluator, startswith='Binary Evaluation Metrics:', sort_value=False)
        print('\n')

        # Multiclasses Evaluation
        per_state_eval = self.__multiclass_evaluator(y_true = table['y_index'].tolist(),
                                                     y_pred = table['y_predict'].tolist(),
                                                     encoding_rules=self.encoding_rules)
        util.print_dict(per_state_eval, startswith='Multiclasses Evaluation Metrics:', sort_value=False)
        # # using MOA Evaluator
        # [self.moa_evaluator.update(act, pred) for act,pred in zip(y_index, y_predict)]

    def __make_stream_pipeline(self):
        # Fusion only works in Edge
        # Step 1: Loading and formatting the relevant dataset
        # For auditing purpose, not all of the dataset must be reviewed
        # We assume the sample for auditing also randomly choosen
        # Auditing purpose should not be used for examining the models or IIDS
        # This solely designed to review the pipeline and runtime
        fs = fusion_engine(base_device = self.base_device,
                                                layer=self.layer,
                                                others=self.device_in_fusion,
                                                load_all=not self.run_audited_version,
                                                sample_size=self.sample_size,
                                                random_seed=self.random_seed)
        feature = fs.feature.copy()
        self.header = fs.header.copy()
        if self.detection_class == 'multiclass':
            target = fs.target.copy()
        else:
            target = iids_base.map_binary_label(ntarget=fs.target.copy())

        if self.run_audited_version:
            iids_base.print_step(step=0, input=[f'{len(target):,}'])
        
        # Encoding the target label into numeric representation
        self.encoded_target, self.encoding_rules = self.__encoding_target_label(ntarget=target)
        # printing running text info
        iids_base.print_step(step=1)
        time.sleep(self.__waiting_time)

        # Step 2: Normalized data if required so
        if self.normalized: 
            scaler = iids_core.__load_scaler_model()
        # printing running text info
        iids_base.print_step(step=2)
        time.sleep(self.__waiting_time)

        # # Step 3: Split dataset into bins, number of instances per bin == window_size
        if self.normalized:
            feature_bin = self.__data_windowing(nlist=feature, window_size=self.window_size)
            scaler.partial_fit(feature_bin[0])
            streamed_ready = scaler.transform(feature_bin[0])
            for bin in feature_bin[1:]:
                scaler.partial_fit(bin)
                new_add = scaler.transform(bin)
                streamed_ready = np.append(streamed_ready, new_add, axis=0)
            # printing running text info
            iids_base.print_step(step=3, input=[f'{len(feature_bin):,}'])
            time.sleep(self.__waiting_time)
        else:
            streamed_ready = feature.copy()
            # printing running text info
            iids_base.print_step(step=3, input=[f'{len(feature_bin):,}'])
            time.sleep(self.__waiting_time)
        
        # Step 4: Preparing dataset as online streaming data
        self.__stream = NumpyStream(X=streamed_ready,
                             y=self.encoded_target)
        iids_base.print_step(step=4)
        time.sleep(self.__waiting_time)
        
    def __make_learner_pipeline(self):
        # Step 5: Setting up learning models
        learner_1, learner_2, learner_3 = iids_base.setup_learning_models(iids_model=self.iids_model,
                                                                        stream=self.__stream,
                                                                        models=self.learning_models,
                                                                        random_seed=self.random_seed)
        self.multi_predict = lambda ins: (learner_1.predict(ins), learner_2.predict(ins), learner_3.predict(ins))
        self.multi_train = lambda ins: (learner_1.train(ins), learner_2.train(ins), learner_3.train(ins))

        iids_base.print_step(step=5)
        time.sleep(self.__waiting_time)
        util.print_list(nlist=self.header, startswith='Feature names:', style='upper')
        self.target_distribution = self.__target_distribution(nlist=self.target)
        util.print_dict(dicts=self.target_distribution, startswith='Target Distribution (%): ', sort_value=True)
        util.nice_print('')

        # Step 6: Starting up test-then-train
    def __ocl_pipeline(self):
        iids_base.print_step(step=6)
        time.sleep(self.__waiting_time)
        instances_seen = 0
        encoded_truth = []
        # we assume that if model predict None, then it is considered as normal (majority class)
        normal_as = self.encoding_rules.get(params.normal_l1label)
        self.__stream.restart()
        while self.__stream.has_more_instances():
            curr_instance = self.__stream.next_instance()
            # first, model to test the current instance
            y_models = list(self.multi_predict(curr_instance))
            y_predict = self.__voting_rules(npredicts=y_models, encoded_normal=normal_as, random_seed=self.random_seed)
            # y_models = output models, final decision, y_true
            output = y_models + [y_predict] + [curr_instance.y_index]
            encoded_truth.append(output)
            # then, model to train the current instance
            self.multi_train(curr_instance)
            instances_seen += 1
            # delete temporary variable to ensure no repeated value
            del y_models, y_predict, output
            # print progress
            if (instances_seen % 1000 == 0) or (self.__stream.has_more_instances() == False):
                progress = instances_seen/self.__stream._len
                msg = util.progress_meter(progress=progress) + f' ~ Instances seen: {instances_seen:,}'
                print(f'>> {msg}', end='\r', flush=True)
        
        # metrics evaluation
        self.__evaluator(model_output=encoded_truth,
                         encoded_normal=normal_as)