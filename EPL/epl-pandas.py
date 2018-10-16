import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd

def wrangle_results(epl_results_dataframe):

    epl_outcomes_dataframe = pd.DataFrame(
        {
            'Season': epl_results_dataframe["Season"],
            'HomeTeam': epl_results_dataframe["HomeTeam"],
            'AwayTeam': epl_results_dataframe["AwayTeam"],
            'FTHG': epl_results_dataframe["FTHG"],
            'FTAG': epl_results_dataframe["FTAG"]
        }
    )

    return epl_outcomes_dataframe

epl_results_dataframe = pd.read_csv("EPL_Set.csv", sep=",")
print(epl_results_dataframe.describe()) # summary stats

print(epl_results_dataframe.head()) # first few records

epl_outcomes_dataframe = wrangle_results(epl_results_dataframe)

print(epl_outcomes_dataframe.head()) # first few records