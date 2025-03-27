import os
from collections import Counter

from matplotlib import pyplot as plt



folder_path = 'train/new-images'
label_counts = Counter([int(f.split('_')[0]) for f in os.listdir(folder_path)])
plt.bar(label_counts.keys(), label_counts.values())
plt.title('Class Distribution')
plt.show()