src1 = 'data_src/TON_IoT/Processed_Linux_dataset/linux_memory1.csv'
src2 = 'data_src/TON_IoT/Processed_Linux_dataset/linux_memory2.csv'
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
df.to_csv('datastream/ToN_IoT_Fog_dataset/linux_memory_feature.csv', sep=',', index=False)