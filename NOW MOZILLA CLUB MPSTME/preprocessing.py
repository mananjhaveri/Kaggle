import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

class prepare:

    def __init__(self):
        return

    def transform(self, df):


        df_ret = df.copy()

        # fill na with median
        df_ret["Trust"] = df_ret["Trust"].fillna(df_ret["Trust"].median())

        # scaling
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer()
        df_ret = scaler.fit_transform(df_ret)

        return df_ret
