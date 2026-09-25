from prepare_data import DATA_PATH, load_clean_data

df = load_clean_data(DATA_PATH)
print(df.shape)