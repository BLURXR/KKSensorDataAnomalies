import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data.csv')
print(df.head())
print(df.info())

df = df.dropna()

sensor_columns = ['CS', 'RP', 'Temperature', 'IP', 'USS']
data = df[sensor_columns]
fail = df["fail"]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

scaled_df = pd.DataFrame(scaled_data, columns=sensor_columns)
scaled_df["fail"] = fail.reset_index(drop=True)
scaled_df.to_csv("scaled_data.csv", index=False)
print("Data Cleaned")
print(scaled_df.head())