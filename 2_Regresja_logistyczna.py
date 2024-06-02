#Importy bibliotek
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('titanic.csv')
print (df.isna().sum()) #suma pustych pól

#Usunięcie kolumn
del (df['ebmarked'])
del (df['class'])
del (df['who'])
del (df['deck'])
del (df['ebmark_town'])
del (df['alive'])
print(df)



#Zamiana sex na dane numeryczne
df = pd.get_dummies(df)

del (df['sex_Female'])

#zmiana nazwy kolumny, True = Male, False = Female
df = df.rename(columns={'sex_Male':'gender'})
print(df)


X = df.iloc[:,[0,1]]
y = df.gender
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

#Skuteczność
print(model.score(X_test, y_test))

#print(pd.DataFrame(confusion_matrix(y_test, model.pr edict(X_test))))

('Sprawdzenie, czy klasy są zablansowane')
print(df.gender.value_counts())

print('Zmiana danych, zrównoważenie klas')
df1 = df.query('gender==0').sample(n=50)
df2 = df.query('gender==1').sample(n=50)
df3 = pd.concat([df1, df2])

X = df3.iloc[:,[0,1]]
y = df3.gender
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

#Skuteczność
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))

print('\nKNN')
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7, weights='uniform')
model.fit(X_train, y_train)

#Skuteczność
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))