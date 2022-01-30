import pandas as pd
from os import listdir
from os.path import isfile, join
folders = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
count = 0
for folder in folders:
    mypath = "COVID-19_Radiography_Dataset/".format(folders[count])
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(len(onlyfiles))
    # for file in onlyfiles:
    # 0 for covid
    # 1 for lung opacity
    # 2 normal
    # 3 ciral pneumonia
    # if
    dict = {'file_name': onlyfiles, 'disease': count}
    df = pd.DataFrame(dict)
    # saving the dataframe
    df.to_csv('file1.csv', index=False, mode='a')
    count += 1
