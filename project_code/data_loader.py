# data_loader.py

import pandas as pd

def load_datasets():
    """
    Charge les datasets depuis les fichiers CSV et les nettoie.
    Returns:
        tuple: Contient les datasets nettoyés pour small et large.
    """
    pathtocsv = 'celeba/'

    csv_s = pathtocsv + 'celeba_buffalo_s.csv'
    csv_l = pathtocsv + 'celeba_buffalo_l.csv'

    try:
        df_s = pd.read_csv(csv_s, engine='python', encoding='utf-8')
        df_l = pd.read_csv(csv_l, engine='python', encoding='utf-8')
    except Exception as e:
        print(f"Error reading CSV files: \n{e}")
        exit()


    # Convert -1 to 0 for binary processing
    df_binary_s = df_s.copy()
    df_binary_l = df_l.copy()

    # Vectorized approach
    df_binary_s.iloc[:, 1:] = (df_binary_s.iloc[:, 1:] == 1).astype(int)
    df_binary_l.iloc[:, 1:] = (df_binary_l.iloc[:, 1:] == 1).astype(int)
    
    df_s_pg1 = df_binary_s.iloc[:, :40].dropna()
    df_l_pg1 = df_binary_l.iloc[:, :40].dropna()
    return df_s_pg1, df_l_pg1
