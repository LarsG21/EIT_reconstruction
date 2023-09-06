from ScioSpec_EIT_Device.data_reader import convert_multi_frequency_eit_to_df

path = "eit_experiments/10_Freq_sweep/20230905 15.59.19/setup_1/setup_1_00001.eit"

df = convert_multi_frequency_eit_to_df(path)

len_before = len(df)

df = df[df["injection_pos"] != df["measuring_electrode"]]
df = df[df["injection_neg"] != df["measuring_electrode"]]

len_after = len(df)

print("Reduced dataframe to factor: ", len_after / len_before)

print(df)
