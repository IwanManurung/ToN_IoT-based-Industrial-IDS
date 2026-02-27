import pandas as pd
import numpy as np
from base import params
from typing import List, Tuple, Literal

class fusion_base:

    @staticmethod
    def map_dist(nlist):
        total = len(nlist)
        dist = {}
        for _ in set(nlist):
            rasio = 100 * nlist.count(_) / total
            dist.update({str(_): rasio})
        return dist
    
    @staticmethod
    def get_sample_distribution(nlist: List,
                                sample_size: float):
        total = len(nlist)
        sample_num = int(sample_size * total)
        dist_int = {}

        for _ in set(nlist):
            if _ != params.normal_l1label:
                num = int(nlist.count(_) / total * sample_num)
                dist_int.update({_: num})

        dist_int.update({params.normal_l1label: sample_num - sum(dist_int.values())})
        return dist_int
    # @staticmethod

class fusion_engine:
    def __init__(self,
                 base_device: str,
                 random_seed: int = 1):
        self.base_device = base_device
        self.random_seed = random_seed
        pass

    def make_edge_dataset(self,
                          all_in_fusion: bool =True,
                          others: List[str] =[],
                          load_all: bool =False,
                          sample_size: float =.1):

        if all_in_fusion:
            ds = pd.read_csv(params.edge_fusion_csv_sources.get(self.base_device), low_memory=False)
        elif all_in_fusion == False:
            ds = pd.read_csv(params.edge_fusion_csv_sources.get(self.base_device), low_memory=False)
            included = params.non_feature.copy()
            for _ in [self.base_device] + others:
                included.extend(params.feature_naming.get(_))
            ds = ds[included].copy()
        
        if load_all:
            feature = ds.drop(columns=params.non_feature).to_numpy()
            target = ds[params.target_label].to_list()
            header = ds.drop(columns=params.non_feature).columns.to_list()
            
        elif load_all == False:
            ds_feature = ds.drop(columns=params.non_feature).to_numpy()
            ds_target = ds[params.target_label].tolist()
            sample_dist = fusion_base.get_sample_distribution(nlist=ds_target,
                                                              sample_size=sample_size)
            
            index_choosen = []
            for key in sample_dist.keys():
                choices = [y for x,y in zip(ds_target, range(len(ds_target))) if x == key]
                rand_gen = np.random.default_rng(self.random_seed)
                choosen = rand_gen.choice(choices, size=sample_dist.get(key))
                index_choosen.extend(choosen)
                del choices, rand_gen, choosen
            
            index_choosen = sorted(index_choosen)
            
            feature = np.array([ds_feature[int(i)].tolist() for i in index_choosen])
            target = [ds_target[int(i)] for i in index_choosen]
            header = ds.drop(columns=params.non_feature).columns.tolist()
        
        return (feature, target, header)
        
    def make_fog_dataset(self,
                          load_all: bool =True,
                          sample_size: float =.2):
        
        ds = pd.read_csv(params.fog_fusion_csv_sources.get(self.base_device), low_memory=False)
        base_feature = ds.drop(columns=params.non_feature).to_numpy()
        base_target = ds[params.target_label].tolist()
        header = ds.drop(columns=params.non_feature).columns.tolist()

        if load_all:    
            return base_feature, base_target, header
        else:
            dist_int = fusion_base.get_sample_distribution(nlist=base_target,
                                                           sample_size=sample_size)
            index_choosen = []
            for key in dist_int.keys():
                choices = [y for x,y in zip(base_target, range(len(base_target))) if x == key]
                rand_gen = np.random.default_rng(self.random_seed)
                choosen = rand_gen.choice(choices, size=dist_int.get(key))
                index_choosen.extend(choosen)
                del choices, rand_gen, choosen            
            index_choosen = sorted(index_choosen)
            feature = np.array([base_feature[i].tolist() for i in index_choosen])
            target = [base_target[i] for i in index_choosen]
            return (feature, target, header)
    