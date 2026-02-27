import pandas as pd
from base import params

class data_fusion_at_edge:
    def __init__(self):
        pass

    def Run_at_once(self):
        for dev in params.edge_device_list:
            base_src = params.csv_sources.get(dev)
            others = [params.csv_sources.get(i) for i in params.edge_device_list if dev != i]
            dst = params.edge_fusion_csv_sources.get(dev)
            base = pd.read_csv(base_src, low_memory=False)
            base_feat = base.drop(columns=[params.target_label]).copy()
            base_col = base.drop(columns=params.non_feature).columns.tolist()
            base_col = [params.ts_label] + base_col
            base_target = base[params.non_feature].copy()
            base_target = base_target[base_target[params.target_label] != params.normal_label].copy()
            base_target = base_target.reset_index(drop=True)
            del base
            for src in others:
                ds = pd.read_csv(src, low_memory=False)
                ds = ds.drop(columns=[params.target_label]).copy()
                base_feat = pd.merge(base_feat, ds, on=params.ts_label, how='left')
                base_feat = base_feat.sort_values(params.ts_label)
                base_feat = base_feat.ffill().bfill().drop_duplicates()
                base_feat = base_feat.groupby(base_col).mean().reset_index()
                del ds
            base_feat = base_feat.dropna().drop_duplicates()
            base = pd.merge(base_feat, base_target, on='ts', how='left')
            base[params.target_label] = base[params.target_label].fillna(params.normal_label)
            base = base.drop_duplicates()
            del base_target, base_feat
            base = base.sort_values(params.ts_label).reset_index(drop=True)
            base.to_csv(dst, sep=',', index=False)
            msg = f"Dataset Edge {dev.upper()} saved to {dst}. Data Length:{len(base):,}."
            print(msg)
            del base, base_src, dst, others