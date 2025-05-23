import pandas as pd
import numpy as np
import os

typecode = 'A321'
lat = 52.04068

file_list = os.listdir('ProcessedFlightData')
file_list = [f for f in file_list if f.endswith('.csv') and typecode in f]

for file in file_list:
    df = pd.read_csv(os.path.join('ProcessedFlightData', file))
    if lat in df['lat'].values:
        idx = df.index[df['lat'] == lat].tolist()
        print(f"{file}: indices {idx}")
        for i in idx:
            print(df["lon"].iloc[i])