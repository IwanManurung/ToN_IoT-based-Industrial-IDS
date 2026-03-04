# import warnings
# warnings.filterwarnings("ignore")

# from src import ICS_IIDS_Supervised
# import json
# import pandas as pd
# from base import params
# from src._iids_caller import ToN_IoT_IIDS

# ids = ToN_IoT_IIDS(layer='edge',
#                    base_device='garage_door',
#                    random_seed=80,
#                    run_audited_version=False)

# proba_threshold = .75
# annot_delay = 300
# ids.run_Anomaly_Detector(prediction_class='normal',
#                          proba_threshold=proba_threshold,)
# ids.run_Supervised(prediction_class='normal')
# ids.run_semi_Supervised_Delay_Label(prediction_class='normal',
#                                     proba_threshold=proba_threshold,
#                                     annot_delay=annot_delay)

# import warnings
# warnings.filterwarnings("ignore")

# from src import ICS_IIDS_Supervised
import json
from base import params
from src._iids_caller import ToN_IoT_IIDS

layer_test = 'edge'
proba_threshold = .75
annot_delay = 300

for dev_under_test in params.edge_device_list:
    ids = ToN_IoT_IIDS(layer=layer_test,
                       window_size=300,
                       base_device=dev_under_test,
                       random_seed=80,
                       run_audited_version=False)

    for class_under_test in ids.available_prediction_class:
        ids.run_Anomaly_Detector(prediction_class=class_under_test,
                                proba_threshold=proba_threshold,)
        fname = f"output/s1{layer_test}-{dev_under_test}-{class_under_test}.json"
        with open(fname, 'w') as f:
            json.dump(fname, f)
        del fname

        ids.run_Supervised(prediction_class=class_under_test)
        fname = f"output/s1{layer_test}-{dev_under_test}-{class_under_test}.json"
        with open(fname, 'w') as f:
            json.dump(fname, f)
        del fname
        ids.run_semi_Supervised_Delay_Label(prediction_class=class_under_test,
                                            proba_threshold=proba_threshold,
                                            annot_delay=annot_delay)
        fname = f"output/s1{layer_test}-{dev_under_test}-{class_under_test}.json"
        with open(fname, 'w') as f:
            json.dump(fname, f)
        del fname