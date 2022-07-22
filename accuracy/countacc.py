import pandas as pd

df = pd.read_csv('Accuracyy.csv')

print(df)

print(df['4'].value_counts())


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


cf_matrix = confusion_matrix(df['0'], df['2'])

print(cf_matrix)

fig, ax= plt.subplots(figsize=(16,9))

ax = sns.heatmap(cf_matrix, annot=True,fmt='d', cmap='Blues',linewidths=.5)
ax.set_title('Confusion Matrix \n\n');
ax.set_xlabel('\nPredicted Result')
ax.xaxis.set_ticks_position('top')
ax.set_ylabel('Actual Result');
ax.xaxis.set_ticklabels(['Business','Entrainnement', 'Tech'])
ax.yaxis.set_ticklabels(['Business','Entrainnement', 'Tech'])


plt.savefig("Accurcay-bast.png")
plt.show()
