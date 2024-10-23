# preprocessor.py
import pandas as pd

class Preprocessor:
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def standardize(self):
        """ Performs Z-score standardization (assume all numeric columns) """
        for key in self.df.keys():
            mu = self.df[key].mean()
            sigma = self.df[key].std()
            self.df[key] = (self.df[key] - mu) / sigma

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

