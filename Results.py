import numpy as np
import cv2 as cv
import time
import pandas as pd

class Results:

    def __init__(self, name, columns: list):
        self.df_name = name
        columns.append('delta_time')
        self.df_result = pd.DataFrame(columns=columns)
        self.timestamp = time.time()

    def add_result(self,data: list):
        timestamp = time.time()
        data.append(timestamp-self.timestamp)
        self.df_result.loc[-1] = data
        # self.df_result.index += 1
        self.df_result.reset_index(inplace=True,drop=True)
        self.timestamp = timestamp

    def save_result(self):
        self.df_result.to_csv(f'results_code/results/{self.df_name}.csv')
        self.df_result.to_parquet(f'results_code/results/{self.df_name}.atc')

    def __str__(self):
        return self.df_result.to_string()

# result = Results('test',['1','2','3'])
# result.add_result(['a','bbb','ccc'])
# result.add_result(['b','bbb','ccc'])
# result.add_result(['c','bbb','ccc'])
# result.add_result(['d','bbb','ccc'])
# result.add_result(['e','bbb','ccc'])
# result.save_result()
# print(result)