src1 = 'data_src/TON_IoT/Processed_Linux_dataset/linux_disk_1.csv'
src2 = 'data_src/TON_IoT/Processed_Linux_dataset/linux_disk_2.csv'
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

df.to_csv('datastream/ToN_IoT_Fog_dataset/linux_disks_feature.csv', sep=',', index=False)