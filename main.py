from cleaning import dataCleaning 
from visuals import visualisation

df = dataCleaning("Dataset.csv")
visualisation(df)
