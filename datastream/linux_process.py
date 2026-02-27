src1 = 'data_src/TON_IoT/Processed_Linux_dataset/Linux_process_1.csv'
src2 = 'data_src/TON_IoT/Processed_Linux_dataset/Linux_process_2.csv'
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
df.to_csv('datastream/ToN_IoT_Fog_dataset/linux_process_feature.csv', sep=',', index=False)