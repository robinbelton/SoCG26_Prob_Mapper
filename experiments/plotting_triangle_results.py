import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('probability_convergence_1k.csv')

fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
axes = axes.flatten()
plt.ylim([0,1.1])
num_points = [3, 9, 18]
gains = [0.6, 0.7,0.8]
probs = []
for points in num_points:
    for gain in gains:
        prob = 1-((4*(1-gain)/(3-(2*gain)))**(points-2))
        probs.append(prob)

print(probs)

for i, column in enumerate(data.columns):
    ax = axes[i]
    data[column].plot(ax=ax, label = '')
    prob = probs[i]
    ax.axhline(y=prob, color='r', linestyle='-', label = 'P(E) = ' + str(round(prob,2)))
    ax.legend()

    ax.set_xlabel("Iteration") # Or another relevant x-axis label
    ax.set_ylabel("Proportion of Graphs that are Triangles")

plt.tight_layout() 
plt.savefig('Fixed Graph 2.pdf', dpi=300)
plt.show()