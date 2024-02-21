import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import column_or_1d


dataset =  pd.read_csv('BreastTissue.csv', names=["Class", "I0", "PA500", "HFS", "DA", "Area", "A/DA", "Max IP", "DR", "P"])
print("Afisam setul de date")
print("\n")
print(dataset)
print("\n")

X=dataset.iloc[0:105, 1:10]
Y=dataset.iloc[0:105, 0:1]

print("Afisam X si Y pentru a verifica delimitarea corecta.")
print("\n")
print(X)
print(Y)
print("\n")

Y = column_or_1d(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,test_size=0.25, random_state=46)

for i in range(-5, 9, 2):
    C=pow(2,i)
    clf= svm.SVC(kernel='linear',C=C)
    clf.fit(X_train, Y_train)
    predictie = clf.predict(X_test)
    
    #Verificam acuratetea pentru setul de test
    acuratete=accuracy_score(Y_test, predictie)
  
    print("\n")
    print("Cost: ", C)
    print("Acuratete: ", acuratete)
    print("\n")
    
    
    
#Afisam seturile utilizate
print("Afisam cele 2 seturi.")
print("Y_test: ", Y_test)
print("Predictie: ", predictie) 
print("\n")