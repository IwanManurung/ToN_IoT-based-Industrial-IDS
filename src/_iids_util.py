from base import params
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from itertools import pairwise
from datastream import fusion_engine
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

class iids_util:

    @staticmethod
    def voting_decision(npredicts, rep_NA=False, rep_NA_as=None):
        # Decision rules for multi classifiers
        # We assumed majority class wins if model is unable to predict or None.
        if rep_NA:
            repNone = lambda x : rep_NA_as if x == None else x
            npredicts = [repNone(x) for x in npredicts]

        if npredicts.count(1) > npredicts.count(0):
            return 1
        else:
            return 0
    
    @staticmethod
    def proba_prediction_rules(nscore: List[float],
                      threshold: float=.5) -> int:
        # Decision rules for multi anomaly detectors
        # Higher score is indicative of Anomaly or Class_1 wins
        assert (threshold > 0.0) & (threshold < 1.0)
        npredicts = [1 if score > threshold else 0 for score in nscore]
        return npredicts
        
    @staticmethod
    def loading_edge_dataset(base_device: str, 
                             all_in_fusion: bool =True, 
                             others: List[str] =[],
                             load_all: bool =False,
                             low_memory: bool =False,
                             sample_size: float =.1,
                             random_seed: int =1):
        
        fs = fusion_engine(base_device=base_device,
                           random_seed=random_seed)
        
        feature, target, header = fs.make_edge_dataset(all_in_fusion=all_in_fusion,
                                                       others=others,
                                                       load_all=load_all,
                                                       sample_size=sample_size)
        
        if low_memory:
            feature = pd.DataFrame(feature.copy(), columns=header)
            for col in feature.columns:
                abs_max = np.abs(feature[col].tolist()).max()
                if abs_max >= 1_000:
                    feature[col] = np.divide(feature[col].tolist(), 1_000_000)
                feature[col] = feature[col].astype(np.float16)
            feature = feature.to_numpy(dtype=np.float16).copy()

        return (feature, target, header)
    
    @staticmethod
    def loading_fog_dataset(base_device: str, 
                            load_all: bool =False,
                            low_memory: bool =False,
                            sample_size: float =.1,
                            random_seed: int =1):
        
        fs = fusion_engine(base_device=base_device,
                           random_seed=random_seed)
        
        feature, target, header = fs.make_fog_dataset(load_all=load_all,
                                                      sample_size=sample_size)
        
        if low_memory:
            feature = pd.DataFrame(feature.copy(), columns=header)
            for col in feature.columns:
                abs_max = np.abs(feature[col].tolist()).max()
                if abs_max >= 1_000:
                    feature[col] = np.divide(feature[col].tolist(), 1_000_000)
                feature[col] = feature[col].astype(np.float16)
            feature = feature.to_numpy(dtype=np.float16).copy()

        return (feature, target, header)

    @staticmethod
    def map_as_binary_class(ntarget: List[str],
                            class_0: str) -> List[int]:
        assert class_0 in list(set(ntarget)), "class_0 must exist in ntarget"
        # Applying Encoding rules: class_0 = 0, rest of the class = 1
        class_map = lambda x: 0 if x == class_0 else 1
        return [class_map(i) for i in ntarget]
    
    @staticmethod
    def load_scaler_model(model: Literal['StandardScaler', 'MinMaxScaler'] ='StandardScaler') -> Callable:
        if model.lower() == 'MinMaxScaler'.lower():
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        return scaler
    
    @staticmethod
    def online_normalization(data: np.ndarray,
                             window_size: int,
                             scaler_model: Literal['StandardScaler', 'MinMaxScaler']) -> np.ndarray:
        if scaler_model == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        bin_range = list(range(0, len(data), int(window_size)))
        bin_range.append(len(data))
        bin = list(pairwise(bin_range))
        bins = [data[start:end] for start,end in bin]
        
        scaler.partial_fit(bins[0])
        scaled_data = scaler.transform(bins[0])
        for bin in bins[1:]:
            scaler.partial_fit(bin)
            new_add = scaler.transform(bin)
            scaled_data = np.append(scaled_data, new_add, axis=0)
            del new_add

        return scaled_data
    
    @staticmethod
    def load_anomaly_model(stream, method, random_seed):
        stream.restart()
        if method.lower() == "OnlineIsolationForest".lower():
            from capymoa.anomaly import OnlineIsolationForest
            detector = OnlineIsolationForest(stream.get_schema(),
                                             window_size=250,
                                             num_trees=100,
                                             random_seed=random_seed)
            
        elif method.lower() == "Autoencoder".lower():
            from capymoa.anomaly import Autoencoder
            num_feat = stream.next_instance().x.shape[0]
            if num_feat <= 3:
                hidden_layer = num_feat - 1
            else:
                hidden_layer = 3
                
            detector = Autoencoder(stream.get_schema(),
                                   random_seed=random_seed,
                                   hidden_layer=hidden_layer)
            
        elif method.lower() == "StreamRHF".lower():
            from capymoa.anomaly import StreamRHF
            detector = StreamRHF(stream.get_schema(),
                                 window_size=250,
                                 num_trees=100,
                                 random_seed=random_seed)
            
        elif method.lower() == "StreamingIsolationForest".lower():
            from capymoa.anomaly import StreamingIsolationForest
            detector = StreamingIsolationForest(stream.get_schema(),
                                                window_size=250,
                                                n_trees=100,
                                                seed=random_seed)
            
        elif method.lower() == "RobustRandomCutForest".lower():
            from capymoa.anomaly import RobustRandomCutForest
            detector = RobustRandomCutForest(stream.get_schema(),
                                             tree_size=250,
                                             n_trees=100,
                                             random_state=random_seed)
            
        elif method.lower() == "AdaptiveIsolationForest".lower():
            from capymoa.anomaly import AdaptiveIsolationForest
            detector = AdaptiveIsolationForest(stream.get_schema(),
                                               window_size=250,
                                               n_trees=100,
                                               seed=random_seed)
        else:
            from capymoa.anomaly import HalfSpaceTrees
            detector = HalfSpaceTrees(stream.get_schema(),
                                      window_size=250,
                                      number_of_trees=100,
                                      random_seed=random_seed)
        return detector
    
    @staticmethod
    def load_classifier_model(stream, method, random_seed):
        stream.restart()
        if method.lower() == 'AdaptiveRandomForestClassifier'.lower():
            from capymoa.classifier import AdaptiveRandomForestClassifier
            classifier = AdaptiveRandomForestClassifier(stream.get_schema(),
                                                        random_seed=random_seed)
        elif method.lower() == 'DynamicWeightedMajority'.lower():
            from capymoa.classifier import DynamicWeightedMajority
            classifier = DynamicWeightedMajority(stream.get_schema(),
                                                 random_seed=random_seed)
        elif method.lower() == 'EFDT'.lower():
            from capymoa.classifier import EFDT
            classifier = EFDT(stream.get_schema(),
                              random_seed=random_seed)
            
        elif method.lower() == 'HoeffdingAdaptiveTree'.lower():
            from capymoa.classifier import HoeffdingAdaptiveTree
            classifier = HoeffdingAdaptiveTree(stream.get_schema(),
                                               random_seed=random_seed)
        elif method.lower() == 'KNN'.lower():
            from capymoa.classifier import KNN
            classifier = KNN(stream.get_schema(),
                             random_seed=random_seed)
            
        elif method.lower() == 'LeveragingBagging'.lower():
            from capymoa.classifier import LeveragingBagging
            classifier = LeveragingBagging(stream.get_schema(),
                                           random_seed=random_seed)
            
        elif method.lower() == 'NaiveBayes'.lower():
            from capymoa.classifier import NaiveBayes
            classifier = NaiveBayes(stream.get_schema(),
                                    random_seed=random_seed)
            
        elif method.lower() == 'OnlineAdwinBagging'.lower():
            from capymoa.classifier import OnlineAdwinBagging
            classifier = OnlineAdwinBagging(stream.get_schema(),
                                            random_seed=random_seed)
            
        elif method.lower() == 'OnlineBagging'.lower():
            from capymoa.classifier import OnlineBagging
            classifier = OnlineBagging(stream.get_schema(),
                                       random_seed=random_seed)
            
        elif method.lower() == 'OnlineSmoothBoost'.lower():
            from capymoa.classifier import OnlineSmoothBoost
            classifier = OnlineSmoothBoost(stream.get_schema(),
                                           random_seed=random_seed)
            
        elif method.lower() == 'OzaBoost'.lower():
            from capymoa.classifier import OzaBoost
            classifier = OzaBoost(stream.get_schema(),
                                  random_seed=random_seed)
            
        elif method.lower() == 'PassiveAggressiveClassifier'.lower():
            from capymoa.classifier import PassiveAggressiveClassifier
            classifier = PassiveAggressiveClassifier(stream.get_schema(),
                                                     random_seed=random_seed)
            
        elif method.lower() == 'SGDClassifier'.lower():
            from capymoa.classifier import SGDClassifier
            classifier = SGDClassifier(stream.get_schema(),
                                       random_seed=random_seed)
            
        elif method.lower() == 'StreamingGradientBoostedTrees'.lower():
            from capymoa.classifier import StreamingGradientBoostedTrees
            classifier = StreamingGradientBoostedTrees(stream.get_schema(),
                                                       random_seed=random_seed)
            
        elif method.lower() == 'StreamingRandomPatches'.lower():
            from capymoa.classifier import StreamingRandomPatches
            classifier = StreamingRandomPatches(stream.get_schema(),
                                                random_seed=random_seed)
        else:
            from capymoa.classifier import HoeffdingTree
            classifier = HoeffdingTree(stream.get_schema(),
                                       random_seed=random_seed)
        return classifier

    @staticmethod
    def setup_learning_models(iids_model, stream, models, random_seed):
        if iids_model.lower() == 'AnomalyModel'.lower():
            learner_1 = iids_util.load_anomaly_model(stream=stream,
                                                     method=models[0],
                                                     random_seed=random_seed)
            learner_2 = iids_util.load_anomaly_model(stream=stream,
                                                     method=models[1],
                                                     random_seed=random_seed)
            learner_3 = iids_util.load_anomaly_model(stream=stream,
                                                     method=models[2],
                                                     random_seed=random_seed)
        elif iids_model.lower() == 'ClassifierModel'.lower():
            learner_1 = iids_util.load_classifier_model(stream=stream,
                                                        method=models[0],
                                                        random_seed=random_seed)
            learner_2 = iids_util.load_classifier_model(stream=stream,
                                                        method=models[1],
                                                        random_seed=random_seed)
            learner_3 = iids_util.load_classifier_model(stream=stream,
                                                        method=models[2],
                                                        random_seed=random_seed)
        return (learner_1, learner_2, learner_3)
    
    @staticmethod
    def evaluation_metrics(y_true: List,
                           y_pred: List) -> Dict[str, float]:
        assert len(y_true) == len(y_pred), 'length of y_true must same as y_pred'
        # list of evaluation metrics
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        kappa = cohen_kappa_score(y1=y_true, y2=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true=y_true, y_pred=y_pred)
        recall = recall_score(y_true=y_true, y_pred=y_pred)

        rounding = lambda x : round(100 * x, 3)

        metrics = {'Accuracy': rounding(acc),
                   'Kappa': round(kappa,2),
                   'F1 score': rounding(f1),
                   'MCC': round(mcc,2),
                   'Precision': rounding(precision),
                   'Recall': rounding(recall)
                   }
        return metrics