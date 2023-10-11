st = time.time()
df = pd.read_csv(f'{data_path}/fra-eng.csv')
dataset = TranslationDataset(df,from_file=False)
et = time.time()
delta = et - st
print(f'elapsed time: {delta:.2f} seconds')


    import timeit

    def load_dataset():
        df = pd.read_csv(f'{data_path}/fra-eng.csv')
        dataset = TranslationDataset(df, from_file=True)

    avg_time = timeit.timeit(load_dataset, number=3) / 3
    print(f"Averaged time for 3 runs: {avg_time:.2f} seconds")