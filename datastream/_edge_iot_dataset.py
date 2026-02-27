# from pathlib import Path
import pandas as pd
src_dir = './data_src/TON_IoT/Processed_IoT_dataset/'
dst_dir = 'datastream/ToN_IoT_Edge_dataset/'

class Cleaning_Edge_Dataset:
    @staticmethod
    def Edge_IoT_Fridge():
        name = "fridge".title()
        src = f"{src_dir}IoT_{name}.csv"
        dst = f"{dst_dir}Edge_IoT_{name}.csv"
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

        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def Edge_IoT_Garage_Door():
        name = "garage_door".title()
        src = f"{src_dir}IoT_{name}.csv"
        dst = f"{dst_dir}Edge_IoT_{name}.csv"
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

        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def Edge_IoT_Modbus():
        name = "modbus".title()
        src = f"{src_dir}IoT_{name}.csv"
        dst = f"{dst_dir}Edge_IoT_{name}.csv"
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

        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def Edge_IoT_Motion_Light():
        name = "motion_light".title()
        src = f"{src_dir}IoT_{name}.csv"
        dst = f"{dst_dir}Edge_IoT_{name}.csv"
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

        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def Edge_IoT_Thermostat():
        name = "thermostat".title()
        src = f"{src_dir}IoT_{name}.csv"
        dst = f"{dst_dir}Edge_IoT_{name}.csv"
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

        df.to_csv(dst, sep=',', index=False)

    @staticmethod
    def Edge_IoT_Weather():
        name = "weather".title()
        src = f"{src_dir}IoT_{name}.csv"
        dst = f"{dst_dir}Edge_IoT_{name}.csv"
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

        df.to_csv(dst, sep=',', index=False)