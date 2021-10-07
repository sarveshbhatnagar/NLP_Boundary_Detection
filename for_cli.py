import pandas as pd
from SBD import WordsFeature


train_file = "SBD.train"

f = pd.read_csv(train_file, sep=r"\s", header=None)
f = f.drop(0, axis=1)

wf = WordsFeature(f)
lst = [wf.get_feature(i) for i in range(len(f))]

print(lst[:10])
