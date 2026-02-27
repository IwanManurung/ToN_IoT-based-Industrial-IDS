import pandas as pd
import glob
from base import params

class cleaning_edge_dataset:
    def __init__(self):
        pass

    def Run_at_once(self):
        self.__Edge_IoT_Fridge()
        self.__Edge_IoT_Garage_Door()
        self.__Edge_IoT_Modbus()
        self.__Edge_IoT_Motion_Light()
        self.__Edge_IoT_Thermostat()
        self.__Edge_IoT_Weather()

    @staticmethod
    def __Edge_IoT_Fridge():
        name = "Fridge"
        src = [i for i in glob.glob(f'{params.raw_edge}*.csv') if name in i][0]
        dst = f"{params.edge_dir}Edge_IoT_{name}.csv"
        df = pd.read_csv(src, low_memory=False)
        df = df.dropna()
        df = df.drop(columns=['label'])
        df = df.rename(columns={'type': 'target'})
        df['target'] = df['target'].str.strip()
        df['temp_condition'] = df['temp_condition'].astype(str)
        df['temp_condition'] = df['temp_condition'].str.strip()
        df['date'] = df['date'].str.strip()
        df['time'] = df['time'].str.strip()

        # encode temp_condition
        att_map = lambda x: 0 if x == 'low' else 1
        df['temp_condition'] = df['temp_condition'].map(att_map)

        # sync time: since date are in AUS tz, offset -07:00
        df['offset'] = '-0700'
        df['dt_str'] = df[['date', 'time', 'offset']].agg(' '.join, axis=1)
        df['ts'] = pd.to_datetime(df['dt_str'], format='%d-%b-%y %H:%M:%S %z')
        df['ts'] = df.ts.values.astype(int)//10**9
        df = df.drop(columns=['date', 'time', 'offset', 'dt_str'])
        df = df.sort_values('ts')
        df = df.drop_duplicates()
        msg = f"Dataset {name} saved into {dst}. Data length: {len(df):,}"
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def __Edge_IoT_Garage_Door():
        name = "Garage_Door"
        src = [i for i in glob.glob(f'{params.raw_edge}*.csv') if name in i][0]
        dst = f"{params.edge_dir}Edge_IoT_{name}.csv"
        df = pd.read_csv(src, low_memory=False)
        df = df.dropna()
        df = df.drop(columns=['label'])
        df = df.rename(columns={'type': 'target'})
        df['target'] = df['target'].str.strip()
        df['sphone_signal'] = df['sphone_signal'].astype(str)
        df['sphone_signal'] = df['sphone_signal'].str.strip()
        df['sphone_signal'] = df['sphone_signal'].str.replace('true', 'on')
        df['sphone_signal'] = df['sphone_signal'].str.replace('false', 'off')
        df['sphone_signal'] = df['sphone_signal'].str.replace('1', 'on')
        df['sphone_signal'] = df['sphone_signal'].str.replace('0', 'off')
        att_map = lambda x: 0 if x == 'off' else 1
        df['sphone_signal'] = df['sphone_signal'].map(att_map)
        del att_map

        df['door_state'] = df['door_state'].astype(str)
        df['door_state'] = df['door_state'].str.strip()
        att_map = lambda x: 0 if x == 'closed' else 1
        df['door_state'] = df['door_state'].map(att_map)
        del att_map

        # sync time: since date are in AUS tz, offset -07:00
        df['offset'] = '-0700'
        df['dt_str'] = df[['date', 'time', 'offset']].agg(' '.join, axis=1)
        df['ts'] = pd.to_datetime(df['dt_str'], format='%d-%b-%y %H:%M:%S %z')
        df['ts'] = df.ts.values.astype(int)//10**9
        df = df.drop(columns=['date', 'time', 'offset', 'dt_str'])
        df = df.sort_values('ts')
        df = df.drop_duplicates()
        msg = f"Dataset {name} saved into {dst}. Data length: {len(df):,}"
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def __Edge_IoT_Modbus():
        name = "Modbus"
        src = [i for i in glob.glob(f'{params.raw_edge}*.csv') if name in i][0]
        dst = f"{params.edge_dir}Edge_IoT_{name}.csv"
        df = pd.read_csv(src, low_memory=False)
        df = df.drop(columns=['label'])
        df = df.rename(columns={'type': 'target'})
        df['target'] = df['target'].str.strip()
        # sync time: since date are in AUS tz, offset -07:00
        df['offset'] = '-0700'
        df['dt_str'] = df[['date', 'time', 'offset']].agg(' '.join, axis=1)
        df['ts'] = pd.to_datetime(df['dt_str'], format='%d-%b-%y %H:%M:%S %z')
        df['ts'] = df.ts.values.astype(int)//10**9
        df = df.drop(columns=['date', 'time', 'offset', 'dt_str'])
        df = df.sort_values('ts')
        df = df.drop_duplicates()
        msg = f"Dataset {name} saved into {dst}. Data length: {len(df):,}"
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def __Edge_IoT_Motion_Light():
        name = "Motion_Light"
        src = [i for i in glob.glob(f'{params.raw_edge}*.csv') if name in i][0]
        dst = f"{params.edge_dir}Edge_IoT_{name}.csv"
        df = pd.read_csv(src, low_memory=False)
        df = df.dropna()
        df = df.drop(columns=['label'])
        df = df.rename(columns={'type': 'target'})
        df['target'] = df['target'].str.strip()
        df['date'] = df['date'].str.strip()
        df['time'] = df['time'].str.strip()

        # light_status
        df['light_status'] = df['light_status'].str.strip()
        att_map = lambda x: 0 if x == 'off' else 1
        df['light_status'] = df['light_status'].map(att_map)
        del att_map
        
        # sync time: since date are in AUS tz, offset -07:00
        df['offset'] = '-0700'
        df['dt_str'] = df[['date', 'time', 'offset']].agg(' '.join, axis=1)
        df['ts'] = pd.to_datetime(df['dt_str'], format='%d-%b-%y %H:%M:%S %z')
        df['ts'] = df.ts.values.astype(int)//10**9
        df = df.drop(columns=['date', 'time', 'offset', 'dt_str'])
        df = df.sort_values('ts')
        df = df.drop_duplicates()
        msg = f"Dataset {name} saved into {dst}. Data length: {len(df):,}"
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def __Edge_IoT_Thermostat():
        name = "Thermostat"
        src = [i for i in glob.glob(f'{params.raw_edge}*.csv') if name in i][0]
        dst = f"{params.edge_dir}Edge_IoT_{name}.csv"
        df = pd.read_csv(src, low_memory=False)
        df = df.dropna()
        df = df.drop(columns=['label'])
        df = df.rename(columns={'type': 'target'})
        df['target'] = df['target'].str.strip()
        df['date'] = df['date'].str.strip()
        df['time'] = df['time'].str.strip()
        
        # sync time: since date are in AUS tz, offset -07:00
        df['offset'] = '-0700'
        df['dt_str'] = df[['date', 'time', 'offset']].agg(' '.join, axis=1)
        df['ts'] = pd.to_datetime(df['dt_str'], format='%d-%b-%y %H:%M:%S %z')
        df['ts'] = df.ts.values.astype(int)//10**9
        df = df.drop(columns=['date', 'time', 'offset', 'dt_str'])
        df = df.sort_values('ts')
        df = df.drop_duplicates()
        msg = f"Dataset {name} saved into {dst}. Data length: {len(df):,}"
        print(msg)
        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def __Edge_IoT_Weather():
        name = "Weather"
        src = [i for i in glob.glob(f'{params.raw_edge}*.csv') if name in i][0]
        dst = f"{params.edge_dir}Edge_IoT_{name}.csv"
        df = pd.read_csv(src, low_memory=False)
        df = df.dropna()
        df = df.drop(columns=['label'])
        df = df.rename(columns={'type': 'target'})
        df['target'] = df['target'].str.strip()
        df['date'] = df['date'].str.strip()
        df['time'] = df['time'].str.strip()
        
        # sync time: since date are in AUS tz, offset -07:00
        df['offset'] = '-0700'
        df['dt_str'] = df[['date', 'time', 'offset']].agg(' '.join, axis=1)
        df['ts'] = pd.to_datetime(df['dt_str'], format='%d-%b-%y %H:%M:%S %z')
        df['ts'] = df.ts.values.astype(int)//10**9
        df = df.drop(columns=['date', 'time', 'offset', 'dt_str'])
        df = df.sort_values('ts')
        df = df.drop_duplicates()
        msg = f"Dataset {name} saved into {dst}. Data length: {len(df):,}"
        print(msg)
        df.to_csv(dst, sep=',', index=False)