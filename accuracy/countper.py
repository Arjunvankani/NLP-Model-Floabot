import pandas as pd
import numpy as np
df = pd.read_csv('Accuracyy.csv')

print(df)

print(df['4'].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


cf_matrix = confusion_matrix(df['0'], df['2'])

print(cf_matrix)


ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Confusion Matrix Perctange\n\n');
ax.set_xlabel('\nPredicted Result')
ax.set_ylabel('Actual Result');
ax.xaxis.set_ticklabels(['Business','Entrainnement', 'Tech'])
ax.yaxis.set_ticklabels(['Business','Entrainnement', 'Tech'])
plt.savefig("Pec-Accurcay-bast.png")

plt.show()