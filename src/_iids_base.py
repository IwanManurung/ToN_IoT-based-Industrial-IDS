from base import util
from base import params
import json
from typing import List

class iids_base:
    def __init__(self):
        pass
    
    @staticmethod
    def map_binary_label(ntarget):
        l1_map = lambda x: params.normal_l1label if x == params.normal_l1label else params.attack_l1label
        return [l1_map(i) for i in ntarget]
    
    @staticmethod
    def binary_map_eval(y: int, normal_state: int):
        '''
        Docstring for binary_map_eval
        
        :param y: mapping items
        :param normal_state: if y == normal_state, return 0, else 1
        '''
        binary = lambda y, normal_state: 0 if y == normal_state else 1 
        return binary(y, normal_state)
    
    @staticmethod
    def print_step(step, input=['']):
        with open (params.saved_configs, 'r') as f:
            cfg = json.load(f)
        space = ' '
        dev = cfg.get('base_device')
        device_in_fusion = cfg.get('device_in_fusion')
        label = cfg.get('base_device')
        layer = cfg.get('layer')
        run_audited_version = cfg.get('run_audited_version')
        seed = cfg.get('random_seed')
        size = cfg.get('window_size')
        normalized = cfg.get('normalized')
        scaler = cfg.get('scaler_model')
        learning_models = cfg.get('learning_models')

        if run_audited_version:
            version = 'Auditing Purpose Only'
        else:
            version = 'Full Dataset'

        if step == 0:
            util.nice_print(f'ToN_IoT-based {layer.upper()} Layer IIDS - {version}')
            util.nice_print(f'Reviewing {input[0]} of Edge IoT {dev.upper()}')
            util.nice_print(f'Random Seed generator: {seed}')
        if step == 1:
            text = ', '.join([i.upper() for i in [dev.upper()] + device_in_fusion])

            text = f'Step {step}: Loading dataset for {text} with {label.upper()} Label: Done'
            util.nice_print(msg=text, align='left', endswith='')
        
        elif step == 2:
            if normalized:
                text = f"Step {step}: Loading sklearn {scaler} model: Done"
            else:
                text = f"Step {step}: Data will be processed without standardization"
            util.nice_print(msg=text, align='left', endswith='')
        
        elif step == 3:
            start = f"Step {step}: "
            text = f"{start}Windowing {input[0]} bins @ {size} instances per bin: Done"
            util.nice_print(msg=text, align='left', endswith='')
            text = f"{space*len(start)}Proceeding online normalization to each bin: Done"
            util.nice_print(msg=text, align='left', endswith='')

        elif step == 4:
            text = "Preparing dataset as online streaming data: Done"
            util.nice_print(msg=text, align='left', endswith='')

        elif step == 5:
            start = f'Step {step}: Setting up learning models:'
            util.nice_print(msg=start, align='left', endswith='')
            for _ in learning_models:
                text = f'{space*len(start)} {_}'
                util.nice_print(msg=text, align='left', endswith='')

        elif step == 6:
            text = f'Step {step}: OCL-based IIDS is now proceeding test-then-train ...'
            util.nice_print(msg=text, align='left', endswith='')

    @staticmethod
    def setup_learning_models(iids_model, stream, models, random_seed):
        if iids_model.lower() == 'AnomalyModel'.lower():
            learner_1 = iids_base.load_anomaly_model(stream=stream, method=models[0], random_seed=random_seed)
            learner_2 = iids_base.load_anomaly_model(stream=stream, method=models[1], random_seed=random_seed)
            learner_3 = iids_base.load_anomaly_model(stream=stream, method=models[2], random_seed=random_seed)
        elif iids_model.lower() == 'ClassifierModel'.lower():
            learner_1 = iids_base.load_classifier_model(stream=stream, method=models[0], random_seed=random_seed)
            learner_2 = iids_base.load_classifier_model(stream=stream, method=models[1], random_seed=random_seed)
            learner_3 = iids_base.load_classifier_model(stream=stream, method=models[2], random_seed=random_seed)
        return (learner_1, learner_2, learner_3)
    
    @staticmethod
    def setup_hybrid_learning_models(stream, anomaly_model, classifier_model, random_seed):
        detector = iids_base.setup_learning_models(iids_model="AnomalyModel",
                                                   stream=stream,
                                                   models=anomaly_model,
                                                   random_seed=random_seed)
        classifier = iids_base.setup_learning_models(iids_model="ClassifierModel",
                                                   stream=stream,
                                                   models=classifier_model,
                                                   random_seed=random_seed)
        return (detector, classifier)

    @staticmethod
    def load_classifier_model(stream, method, random_seed):
        if method.lower() == 'AdaptiveRandomForestClassifier'.lower():
            from capymoa.classifier import AdaptiveRandomForestClassifier
            classifier = AdaptiveRandomForestClassifier(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'DynamicWeightedMajority'.lower():
            from capymoa.classifier import DynamicWeightedMajority
            classifier = DynamicWeightedMajority(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'EFDT'.lower():
            from capymoa.classifier import EFDT
            classifier = EFDT(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'HoeffdingAdaptiveTree'.lower():
            from capymoa.classifier import HoeffdingAdaptiveTree
            classifier = HoeffdingAdaptiveTree(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'KNN'.lower():
            from capymoa.classifier import KNN
            classifier = KNN(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'LeveragingBagging'.lower():
            from capymoa.classifier import LeveragingBagging
            classifier = LeveragingBagging(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'NaiveBayes'.lower():
            from capymoa.classifier import NaiveBayes
            classifier = NaiveBayes(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'OnlineAdwinBagging'.lower():
            from capymoa.classifier import OnlineAdwinBagging
            classifier = OnlineAdwinBagging(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'OnlineBagging'.lower():
            from capymoa.classifier import OnlineBagging
            classifier = OnlineBagging(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'OnlineSmoothBoost'.lower():
            from capymoa.classifier import OnlineSmoothBoost
            classifier = OnlineSmoothBoost(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'OzaBoost'.lower():
            from capymoa.classifier import OzaBoost
            classifier = OzaBoost(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'PassiveAggressiveClassifier'.lower():
            from capymoa.classifier import PassiveAggressiveClassifier
            classifier = PassiveAggressiveClassifier(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'SGDClassifier'.lower():
            from capymoa.classifier import SGDClassifier
            classifier = SGDClassifier(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'StreamingGradientBoostedTrees'.lower():
            from capymoa.classifier import StreamingGradientBoostedTrees
            classifier = StreamingGradientBoostedTrees(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'StreamingRandomPatches'.lower():
            from capymoa.classifier import StreamingRandomPatches
            classifier = StreamingRandomPatches(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == 'HoeffdingTree'.lower():
            from capymoa.classifier import HoeffdingTree
            classifier = HoeffdingTree(stream.get_schema(), random_seed=random_seed)
        else:
            from capymoa.classifier import HoeffdingTree
            classifier = HoeffdingTree(stream.get_schema(), random_seed=random_seed)
        return classifier
    
    def load_anomaly_model(stream, method, random_seed):
        if method.lower() == "OnlineIsolationForest".lower():
            from capymoa.anomaly import OnlineIsolationForest
            detector = OnlineIsolationForest(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == "Autoencoder".lower():
            from capymoa.anomaly import Autoencoder
            detector = Autoencoder(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == "StreamRHF".lower():
            from capymoa.anomaly import StreamRHF
            detector = StreamRHF(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == "StreamingIsolationForest".lower():
            from capymoa.anomaly import StreamingIsolationForest
            detector = StreamingIsolationForest(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == "RobustRandomCutForest".lower():
            from capymoa.anomaly import RobustRandomCutForest
            detector = RobustRandomCutForest(stream.get_schema(), random_seed=random_seed)
        elif method.lower() == "AdaptiveIsolationForest".lower():
            from capymoa.anomaly import AdaptiveIsolationForest
            detector = AdaptiveIsolationForest(stream.get_schema(), random_seed=random_seed)
        else:
            from capymoa.anomaly import HalfSpaceTrees
            detector = HalfSpaceTrees(stream.get_schema(), random_seed=random_seed)
        return detector