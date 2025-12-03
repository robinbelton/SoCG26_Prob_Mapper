import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('probability_convergence_square_1k.csv')

fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()
plt.ylim([0,1.1])

for i, column in enumerate(data.columns):
    ax = axes[i]
    data[column].plot(ax=ax, label = '')

    ax.set_xlabel("Iteration") 
    ax.set_ylabel("Proportion of Graphs that contain a Square")

plt.tight_layout() 
plt.savefig('Fixed Graph 3.pdf', dpi=300)
plt.show()