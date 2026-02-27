import warnings
warnings.filterwarnings("ignore")

from src import ICS_IIDS_Delay_Label
import json
import pandas as pd

def get_available_classes(dev):
    src = params.csv_sources.get(dev)
    gt = pd.read_csv(src)
    return list(gt['target'].unique())
    
run_audited_version = False

iids = ICS_IIDS_Delay_Label(
    anomaly_models=['Autoencoder', 'HalfSpaceTrees', 'AdaptiveIsolationForest'],
    classifier_models=['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree'],
    delayed_label=1_000,
    normalized=True,
    scaler_model='StandardScaler',
    window_size=60,
    random_seed=80,
    run_audited_version=run_audited_version,
    audited_size=.1)

for dev_under_test in params.edge_device_list:
    available_classes = get_available_classes(dev=dev_under_test)
    for class_under_test in available_classes:
        iids.Edge_Layer_IIDS(base_device=dev_under_test,
                                all_in_fusion=True,
                                prediction_class=class_under_test)
        
        prefix = f'output/Edge-IIDS_S3_{dev_under_test}_{class_under_test}'
        
        iids.model_output.to_csv(f'{prefix}.csv', sep=',', index=False)
        
        with open(f'{prefix}.json', 'w') as file:
            json.dump(iids.metrics, file)
        
        del prefix
        
        iids.Fog_Layer_IIDS(base_device=dev_under_test,
                            prediction_class=class_under_test)
            
        # save model_output and metrics
        prefix = f'output/Fog-IIDS_S3_{dev_under_test}_{class_under_test}'
        
        iids.model_output.to_csv(f'{prefix}.csv', sep=',', index=False)
        
        with open(f'{prefix}.json', 'w') as file:
            json.dump(iids.metrics, file)
        
        del prefix