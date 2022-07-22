import intent_inference_containers 
import pandas as pd
import csv
bert = intent_inference_containers.BertHuggingfaceIntentContainer()

c = 0
n = 0
r = 0
br = 0
tr = 0
er = 0
blr = 0
tlr = 0
elr = 0
bwr = 0
twr = 0
ewr = 0
main = []

with open('final-modelbest.csv',encoding="utf-8") as file_obj:
    heading = next(file_obj)
    reader_obj = csv.reader(file_obj)
    for row in reader_obj:
        print("Query:" + str(c) + "|No:" + str(n) + "|Wrong:" + str(r))
        print('__________________________')
        #print(row[0])
        result, conf = bert.Response('final-modelbest',query=row[0])[0]
        
        com = row[1][:1]+result[:1]
        
        
        re = [row[1],row[0], result, conf,com]
        main.append(re)
        #print(main)
        

        if result == row[1]:
            if conf > 0.5:
                
                print("Yes")
                print(row[1] +'*'+ result)
                print(result)
                
                if (result) == 'business':
                    br += 1
                if (result) == 'tech':
                    tr += 1
                if (result) == 'entertainment':
                    er += 1  
                c += 1
            
            else:
                
                print("No")
                print(row[1] +'*'+ result)
                if result == 'business':
                    blr += 1
                if result == 'tech':
                    tlr += 1
                if result == 'entertainment':
                    elr += 1 
                n += 1   
        else:
            
            print("Wrong")
            print(row[1] +'*'+ result)
            if result == 'business':
                    bwr += 1
            if result == 'tech':
                    twr += 1
            if result == 'entertainment':
                    ewr += 1 
            r += 1
print(main)   
df = pd.DataFrame(main)
print(df[4].value_counts())    
df.to_csv('Accuracybest.csv', mode='a')
print("Result ")
print("Query : "+str(c))
print("NO : "+str(n))
print("WRong :"+str(r))
print("Total Q :" + str(c+n+r))
print("Acc :" + str((c)/(c+n+r)))
print('--------------------')
print("Catogery : ")
print("Business:" + str(br) + "|Tech:" + str(tr) + "|Entertainment:" + str(er))
print('--------------------')
print("Low confidence Catogery: ")
print("Low Business:" + str(blr) + "|Low Tech:" + str(tlr) + "|Low Entertainment:" + str(elr))
print('--------------------')
print("Wrong Catogery: ")
print("Wrong Business:" + str(bwr) + "|Wrong Tech:" + str(twr) + "|Wrong Entertainment:" + str(ewr))
print("__________________________________")
print("__________________________________")
print("Total Query " + "Business:" + str(br+blr+bwr) + "|Tech:" + str(tr+tlr+twr) + "|Entertainment:" + str(er+elr+ewr))
print("Total Right Query " + "Business:" + str(br+blr) + "|Tech:" + str(tr+tlr) + "|Entertainment:" + str(er+elr))
print("Total Wrong Query " + "Business:" + str(bwr) + "|Wrong Tech:" + str(twr) + "|Wrong Entertainment:" + str(ewr))
print("__________________________________")
print("__________________________________")
#print("Total Accuracy"+ "Business:" + str((br)/(br+blr+bwr)) + "|Tech:" + str((tr)/(tr+tlr+twr)) + "|Entertainment:" + str(er)/(er+elr+ewr))
