from collections import Counter
from os import walk
from typing import Dict
from keras.models import Model
from matplotlib import pyplot as plt

filenames = "".join([x[0:5] for x in next(walk("./bad_predictions"))[2]])
counter = Counter(filenames)

for x in sorted(counter):
    print(f"{x} -> {counter[x]}")

data: Dict[str, int] = dict({x: counter[x] for x in sorted(counter)})

m: Model

plt.title("Character distribution")
plt.bar(data.keys(), data.values())
plt.show()
