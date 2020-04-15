import os.path
from os import listdir
import pandas as pd
import numpy as np

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("Covid reader")

with open("covid_index.txt") as f:
    content = f.readlines()


def filter_content(content, filter):
    response = []
    for idx, i in enumerate(content):
        if filter.lower() in i.split()[-1].lower():
            for s in i.split():
                if any(ext in s for ext in [".jpeg", ".jpg", ".png"]):
                    response.append(s)
    return response

covid_data = filter_content(content, "COVID-19")
normal_data = filter_content(content, "normal")
print(len(normal_data))
pneumonia_data = filter_content(content, "pneumonia")
print(len(pneumonia_data))

covidx_paths = ["/nfs/managed_datasets/COVID19/XRAY/covidx_dataset/raw_data/train", "/nfs/managed_datasets/COVID19/XRAY/covidx_dataset/raw_data/test"]
counter = 0

covid_positive = []
for image in covid_data:
    for pat in covidx_paths:
        ft = os.path.join(pat, image)
        if os.path.isfile(ft):
            covid_positive.append(ft)
print("Covid positive examples", len(covid_positive))

normal = []
for image in normal_data:
    for pat in covidx_paths:
        ft = os.path.join(pat, image)
        if os.path.isfile(ft):
            normal.append(ft)
print("Normal examples", len(normal))

pneumonia = []
for image in pneumonia_data:
    for pat in covidx_paths:
        ft = os.path.join(pat, image)
        if os.path.isfile(ft):
            pneumonia.append(ft)
print("Pneumonia positive examples", len(pneumonia))

usf_paths = ["/nfs/managed_datasets/COVID19/XRAY/usf_dataset/train2/covid", "/nfs/managed_datasets/COVID19/XRAY/usf_dataset/validation2/covid"]
for pat in usf_paths:
    covid_positive.extend([os.path.join(pat, f) for f in os.listdir(pat) if os.path.isfile(os.path.join(pat, f))])

print("Covid positive examples", len(covid_positive))
with open("covid_positive.txt", "a", newline='\n') as fp:
    fp.write("\n".join(covid_positive))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("Chexpert reader")
chexpertpath = "/nfs/managed_datasets/chexpert/CheXpert-v1.0-small"
traincsv = os.path.join(chexpertpath, "train.csv")
validcsv = os.path.join(chexpertpath, "valid.csv")

df = pd.read_csv(traincsv)
df = df.fillna(0)
df.describe().to_csv("chexpert_description.csv")
# labels = []
names = df.columns.values[5:]
print("Loading {1} pathologies from Chexpert: {0}".format(names, len(names)))
print("Selecting only Frontal views")


with open("{}_positive.txt".format("Pneumonia"), "a", newline='\n') as fp:
    for item in pneumonia:
        fp.write("%s\n" % item)
with open("{}_positive.txt".format("No Finding"), "a", newline='\n') as fp:
    for item in normal:
        fp.write("%s\n" % item)


counter = 0
for i, row in df.iterrows():
    if row[3] == "Frontal":
        label = np.abs(row[5:].values) # convert uncertain -1 to positive 1
        positive = np.where(label == 1.0)[0]
        if len(positive) == 1:
            with open("{}_positive.txt".format(names[positive[0]]), "a", newline='\n') as fp:
                counter += 1
                fp.write(os.path.join(chexpertpath, *row['Path'].split("/")[1:]))
                fp.write('\n')
print("Wrote {} examples in total".format(counter))



import glob
for name in glob.glob("*_positive.txt"):
    number_of_lines = len(open(name).readlines())
    print("{} positive examples {}".format(name.split("_positive.txt")[0], number_of_lines))