import pandas as pd

# read an excel file and convert
# into a dataframe object
df = pd.DataFrame(pd.read_excel("WHO_FAQ.xlsx"))
df.to_csv("Test.csv",
          index=False,
          header=True, sep="&")
