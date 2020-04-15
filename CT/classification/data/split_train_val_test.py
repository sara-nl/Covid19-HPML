import os
import shutil


textfiles = [x for x in os.listdir('.') if '.txt' in x]

for textfile in textfiles:
    with open(textfile, 'r') as f:
        files = f.readlines()

    files = [x.strip() for x in files]

    label = textfile.split('_')[1].replace('.txt', '')
    split = textfile.split('_')[0]

    os.makedirs(os.path.join(split, label), exist_ok=True)

    for file in files:
        path_to_file = os.path.join(label, file)
        shutil.move(path_to_file, os.path.join(split, label))
