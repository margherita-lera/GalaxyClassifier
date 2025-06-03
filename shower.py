import matplotlib.pyplot as plt
import pickle
import sys

with open(f'loss_{sys.argv[1]}.pickle', 'rb') as fin: losses = pickle.load(fin)
train_score = losses['epoch']
val_score = losses['vepoch']


fig, ax = plt.subplots(figsize=(8, 8))
ax.grid(alpha=.3)
ax.axhline(0.087, ls='--', c='black', label='Early Stopping')
ax.plot(range(len(train_score)), train_score, label='Training Loss', color='black')
ax.plot(range(len(val_score)), val_score, label='Validation Loss', color='red')
ax.set_xlabel('Epochs', fontsize=16)
ax.set_ylabel('Loss', fontsize=16)
ax.set_title('Training', fontsize=22, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(fontsize=16, loc='upper right')

plt.show()
#Val200: 0.06211485574330829
#Train200: 0.06037804431027529

