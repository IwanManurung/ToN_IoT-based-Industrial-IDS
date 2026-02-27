from base import params
import pandas as pd

class data_fusion_at_fog:
    def __init__(self):
        pass
    
    @staticmethod
    def __get_groundtruth(dev):
        # loading groundtruth from Edge Datasets
        src = params.edge_fusion_csv_sources.get(dev)
        gt = pd.read_csv(src, low_memory=False)
        colname = params.non_feature + params.feature_naming.get(dev)
        gt = gt[colname].copy()
        gt = gt.sort_values(params.ts_label).copy()
        return gt
    
    def Run_at_once(self):
        fog_memory = pd.read_csv(params.csv_sources.get('linux_memory'), low_memory=False)
        fog_disk = pd.read_csv(params.csv_sources.get('linux_disks'), low_memory=False)
        fog_process = pd.read_csv(params.csv_sources.get('linux_process'), low_memory=False)

        for dev in params.edge_device_list:
            edge_src = params.csv_sources.get(dev)
            base = self.__get_groundtruth(dev=dev)
            base = pd.merge(base, fog_disk, on=params.ts_label, how='left')
            base = pd.merge(base, fog_memory, on=params.ts_label, how='left')
            base = pd.merge(base, fog_process, on=params.ts_label, how='left')
            base = base.ffill()
            base = base.bfill()
            colname = params.non_feature + params.feature_naming.get(dev)
            base = base.groupby(colname).mean().reset_index()
            base = base.dropna().drop_duplicates().sort_values(params.ts_label).reset_index(drop=True)
            base = base.drop(columns=params.feature_naming.get(dev)).copy()
            fname = params.fog_fusion_csv_sources.get(dev)
            base.to_csv(fname, sep=',', index=False)
            name = dev.replace('_', ' ').title()
            msg = f"Fog Dataset for {name} with its Edge Ground Truth is created. Data length: {len(base):,}"
            print(msg)
            del edge_src, base, fname