import warnings
warnings.filterwarnings("ignore")

from src import ICS_IIDS_Supervised
import json
import pandas as pd

run_audited_version = True
dev_under_test = 'fridge'
class_under_test = 'normal'
iids_layer = 'edge'

iids = ICS_IIDS_Supervised(
    classifier_models=['DynamicWeightedMajority', 'OzaBoost', 'HoeffdingAdaptiveTree'],
    normalized=True,
    scaler_model='StandardScaler',
    window_size=60,
    random_seed=80,
    run_audited_version=run_audited_version,
    audited_size=.1)

iids_layer == 'edge'
iids.Edge_Layer_IIDS(base_device=dev_under_test,
                        all_in_fusion=True,
                        prediction_class=class_under_test)
# save model_output and metrics
prefix = f'output/{iids_layer.title()}-IIDS_S2_{dev_under_test}_{class_under_test}'

iids.model_output.to_csv(f'{prefix}.csv', sep=',', index=False)

with open(f'{prefix}.json', 'w') as file:
    json.dump(iids.metrics, file)
del iids_layer, prefix

iids_layer = 'fog'
iids.Fog_Layer_IIDS(base_device=dev_under_test,
                    prediction_class=class_under_test)
    
# save model_output and metrics
prefix = f'output/{iids_layer.title()}-IIDS_S2_{dev_under_test}_{class_under_test}'

iids.model_output.to_csv(f'{prefix}.csv', sep=',', index=False)

with open(f'{prefix}.json', 'w') as file:
    json.dump(iids.metrics, file)
del iids_layer, prefix