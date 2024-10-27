# preprocessor.py
import pandas as pd

class Preprocessor:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def standardize(self):
        self.df["F1"] = (self.df["F1"] - 69) / 14
        self.df["F2"] = (self.df["F2"] - 1.4) / 1
        self.df["F3"] = (self.df["F3"] - 4.5) / 3.5
        self.df["F4"] = (self.df["F4"] - 37) / 1.1
        self.df["F5"] = (self.df["F5"] - 130) / 23
        self.df["F6"] = (self.df["F6"] - 72) / 15
        self.df["F7"] = (self.df["F7"] - 102) / 20
        self.df["F8"] = (self.df["F8"] - 22) / 5
        self.df["F9"] = (self.df["F9"] - 138) / 6
        self.df["F10"] = (self.df["F10"] - 4) / 1.2
        self.df["F11"] = (self.df["F11"] - 2.5) / 2
        self.df["F12"] = (self.df["F12"] - 31.5) / 10
        self.df["F13"] = (self.df["F13"] - 14) / 10
        self.df["F14"] = (self.df["F14"] - 12) / 3.9
        self.df["F15"] = (self.df["F15"] - 38) / 17
        self.df["F16"] = (self.df["F16"] - 185) / 110
        self.df["F17"] = (self.df["F17"] - 2.2) / 2.5

    def feature_selection(self):
        del self.df["F7"]
        del self.df["F11"]
        del self.df["F17"]

    def remove_index(self):
        del self.df[self.df.columns[0]]  
    
    def fillna(self):
        for key in self.df.keys():
            if self.df[key].dtype.name == "object":
                # Categorical data
                replacement = self.df[key].mode(False)[0]
                self.df[key] = self.df[key].fillna(replacement)
            else:
                # Numerical data
                replacement = self.df[key].mean()
                self.df[key] = self.df[key].fillna(replacement)

    def yesno_to_int(self):
        for key in self.df.keys():
            if self.df[key].dtype.name == "object":
                # All categorical data is Yes/No
                self.df[key] = self.df[key].replace(
                    ["Yes", "Female", "Infected",
                     "No", "Male", "Non-infected"], 
                    ["1", "1", "1",
                     "0", "0", "0"]
                ).astype(int)

