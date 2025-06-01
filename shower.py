import matplotlib.pyplot as plt
import pickle
import sys

with open(f'loss_{sys.argv[1]}.pickle', 'rb') as fin: losses = pickle.load(fin)
train_score = losses['epoch']
val_score = losses['vepoch']


fig, ax = plt.subplots(figsize=(8, 8))
ax.grid(alpha=.3)
ax.axhline(.062, lw=1, ls='--', c='black')
ax.plot(range(len(train_score)), train_score, label='Training')
ax.plot(range(len(val_score)), val_score, label='Validation')
ax.legend(loc='upper right')

plt.show()
#Val200: 0.06211485574330829
#Train200: 0.06037804431027529
