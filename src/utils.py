import pandas as pd


def detect_frequency(df):

    inferred = pd.infer_freq(df["ds"])

    if inferred is None:
        return "D"

    return inferred