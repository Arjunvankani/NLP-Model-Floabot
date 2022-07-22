import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


df = pd.read_csv('Accuracyy.csv')



print(df['4'].value_counts())
print(accuracy_score(df['0'], df['2']))
cf_matrix = confusion_matrix(df['0'], df['2'])
print(cf_matrix)
print(classification_report(df['0'], df['2']))


exit(0)

bb = cf_matrix[0][0]
be = cf_matrix[0][1]
bt = cf_matrix[0][2]
eb = cf_matrix[1][0]
ee = cf_matrix[1][1]
et = cf_matrix[2][2]
tb = cf_matrix[2][0]
te = cf_matrix[2][1]
tt = cf_matrix[2][2]
total = sum(cf_matrix[0] + cf_matrix[1]+ cf_matrix[2])

print('cf_matrix[0][0] : bb - '+str(cf_matrix[0][0]))
print('cf_matrix[0][1] : be - '+str(cf_matrix[0][1]))
print('cf_matrix[0][2] : bt - '+str(cf_matrix[0][2]))
print('cf_matrix[1][0] : eb - '+str(cf_matrix[1][0]))
print('cf_matrix[1][1] : ee - '+str(cf_matrix[1][1]))
print('cf_matrix[1][2] : et - '+str(cf_matrix[2][2]))
print('cf_matrix[2][0] : tb - '+str(cf_matrix[2][0]))
print('cf_matrix[2][1] : te - '+str(cf_matrix[2][1]))
print('cf_matrix[2][2] : tt - '+str(cf_matrix[2][2]))

print("Accurcay : "+ str((bb+ee+tt)/total))
# precision = tp / tp+fp


pb = (bb)/(bb+be+bt)
pe = (ee)/(ee+eb+et)
pt = (tt)/(tt+tb+te)

print("Pricision of Business: "+ str(pb))
print("Pricision of Entrainnement: "+ str(pe))
print("Pricision of Tech: "+ str(pt))

# reacall
rb = (bb)/(bb+eb+tb)
re = (ee)/(ee+be+te)
rt = (tt)/(tt+bt+et)


print("Recall of Business: "+ str(rb))
print("Recall of Entrainnement: "+ str(re))
print("Recall of Tech: "+ str(rt))

# f1 score  = 2 * (precision * recall) / (precision + recall)

print("F1  of Business: "+ str())
print("F1 of Entrainnement: "+ str())
print("F1 of Tech: "+ str())
