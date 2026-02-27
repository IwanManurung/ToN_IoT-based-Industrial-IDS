df = pd.read_csv('data_src/TON_IoT/Processed_Windows_dataset/windows10_dataset.csv', low_memory=False)
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

df.to_csv('datastream/ToN_IoT_Fog_dataset/windows10.csv', sep=',', index=False)