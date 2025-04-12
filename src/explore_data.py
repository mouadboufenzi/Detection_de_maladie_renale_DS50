def print_basic_info(df):
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())
    print("\nDescription:")
    print(df.describe(include='all'))
