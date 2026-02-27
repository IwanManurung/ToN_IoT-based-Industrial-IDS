import pandas as pd
import numpy as np
from base import params

class cleaning_fog_dataset:
    def __init__(self):
        pass

    def Run_at_once(self):
        self.__Fog_Linux_Disk()
        self.__Fog_Linux_Memory()
        self.__Fog_Linux_Process()
        self.__Fog_Linux_Windows7()
        self.__Fog_Linux_Windows10()

    @staticmethod
    def __Fog_Linux_Disk():
        src1 = f'{params.raw_linux}linux_disk_1.csv'
        src2 = f'{params.raw_linux}linux_disk_2.csv'
        dst = f'{params.fog_dir}linux_disks_feature.csv'
        df1 = pd.read_csv(src1, low_memory=False)
        df1 = df1.drop(columns=['PID','CMD', 'label', 'type'])
        df1 = df1.drop_duplicates()
        df2 = pd.read_csv(src2, low_memory=False)
        df2 = df2.drop(columns=['PID', 'CMD', 'label', 'type'])
        df2 = df2.drop_duplicates()
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates()
        del df1, df2

        col = 'RDDSK'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].str.replace('.4M', '400000')
        df[col] = df[col].str.replace('.3M', '300000')
        df[col] = df[col].str.replace('.8M', '800000')
        df[col] = df[col].str.replace('.9M', '900000')
        df[col] = df[col].astype(float)

        col = 'WRDSK'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(' ', '')
        df[col] = df[col].str.replace('-', '')
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].str.replace('.2M', '200000')
        df[col] = df[col].str.replace('.5M', '500000')
        df[col] = df[col].str.replace('.3M', '300000')
        df[col] = df[col].str.replace('.0M', '00000')
        df[col] = df[col].str.replace('', '0')
        df[col] = df[col].astype(float)

        col = 'WCANCL'
        df[col] = df[col].astype(str)
        index_todrop = []
        for i in df.index:
            val = df.loc[i, col]
            try:
                float(val)
            except:
                if '%' in val:
                    index_todrop.append(i)
        df = df.drop(index=index_todrop)
        del index_todrop
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].astype(float)

        col = 'DSK'
        df[col] = df[col].astype(str)
        index_todrop = []
        for i in df.index:
            val = df.loc[i, col]
            try:
                float(val)
            except:
                if ('%' in val) or ('worer' in val) or ('atop' in val) or ('dhclient' in val) or ('apache2' in val):
                    index_todrop.append(i)
        df = df.drop(index=index_todrop)
        del index_todrop
        df[col] = df[col].astype(float)

        df = df.dropna().drop_duplicates().sort_values('ts').reset_index(drop=True)
        msg = f'Fog dataset: Linux Disk is ready. Data length: {len(df):,}'
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    def __Fog_Linux_Memory():
        src1 = f'{params.raw_linux}linux_memory1.csv'
        src2 = f'{params.raw_linux}linux_memory2.csv'
        dst = f'{params.fog_dir}linux_memory_feature.csv'
        df1 = pd.read_csv(src1, low_memory=False)
        df1 = df1.drop(columns=['PID','CMD', 'label', 'type'])
        df1 = df1.drop_duplicates()
        df2 = pd.read_csv(src2, low_memory=False)
        df2 = df2.drop(columns=['PID', 'CMD', 'label', 'type'])
        df2 = df2.drop_duplicates()
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates()
        del df1, df2

        col = 'MINFLT'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].astype(float)
        col = 'MAJFLT'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('M', '000000')
        df[col] = df[col].astype(float)
        col = 'VSTEXT'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].astype(float)
        col = 'RSIZE'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].str.replace('.4M', '400000')
        df[col] = df[col].str.replace('.2M', '400000')
        df[col] = df[col].astype(float)
        col = 'VGROW'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace(' ', '')
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].astype(float)
        col = 'RGROW'
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('K', '000')
        df[col] = df[col].astype(float)

        df = df.dropna().drop_duplicates().sort_values('ts').reset_index(drop=True)
        msg = f'Fog dataset: Linux Memory is ready. Data length: {len(df):,}'
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    def __Fog_Linux_Process():
        src1 = f'{params.raw_linux}Linux_process_1.csv'
        src2 = f'{params.raw_linux}Linux_process_2.csv'
        dst = f'{params.fog_dir}linux_process_feature.csv'
        df1 = pd.read_csv(src1, low_memory=False)
        df1 = df1.drop(columns=['PID','CMD', 'label', 'type'])
        df1 = df1.drop_duplicates()
        df2 = pd.read_csv(src2, low_memory=False)
        df2 = df2.drop(columns=['PID', 'CMD', 'label', 'type'])
        df2 = df2.drop_duplicates()
        df = pd.concat([df1, df2], ignore_index=True)
        df = df.drop_duplicates()
        del df1, df2

        # drop colum POLI, State, and Status
        df = df.drop(columns=['POLI', 'Status', 'State'])

        df = df.dropna().drop_duplicates().sort_values('ts').reset_index(drop=True)

        msg = f'Fog dataset: Linux Process is ready. Data length: {len(df):,}'
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    def __Fog_Linux_Windows7():
        src = f'{params.raw_windows}windows7_dataset.csv'
        dst = f'{params.fog_dir}windows7.csv'
        df = pd.read_csv(src, low_memory=False)
        df = df.drop(columns=['label', 'type'])
        err_col = []
        for col in df.columns.tolist():
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(' ','')
            df[col] = df[col].str.replace('','0')
            try:
                df[col] = df[col].astype(float)
            except:
                print(col)
                err_col.append(col)
        for col in df.columns.tolist():
            if df[col].max() > 1.0:
                df[col] = np.round(df[col].tolist(), 3)

        msg = f'Fog dataset: Windows7 is ready. Data length: {len(df):,}'
        print(msg)

        df.to_csv(dst, sep=',', index=False)
    
    def __Fog_Linux_Windows10():
        src = f'{params.raw_windows}windows10_dataset.csv'
        dst = f'{params.fog_dir}windows10.csv'
        df = pd.read_csv(src, low_memory=False)
        df = df.drop(columns=['label', 'type'])
        err_col = []
        for col in df.columns.tolist():
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace(' ','')
            df[col] = df[col].str.replace('','0')
            df[col] = df[col].str.replace('E0','E')
            try:
                df[col] = df[col].astype(float)
            except:
                print(col)
                err_col.append(col)
        for col in df.columns.tolist():
            if df[col].max() > 1.0:
                df[col] = np.round(df[col].tolist(), 3)

        msg = f'Fog dataset: Windows10 is ready. Data length: {len(df):,}'
        print(msg)
        df.to_csv(dst, sep=',', index=False)