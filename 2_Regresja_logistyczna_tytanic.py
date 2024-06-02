# Importy bibliotek
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# Zaczytanie pliku z danymi
df = pd.read_csv('titanic.csv')
# Usunięcie kolumn
del (df['embarked'])
del (df['class'])
del (df['who'])
del (df['deck'])
del (df['embark_town'])
del (df['alive'])
# Zamiana wartości na dane numeryczne
df = pd.get_dummies(df)
# Usunięcie nadmiarowej kolumny
del (df['sex_female'])
# Zmiana nazwy kolumny
df = df.rename(columns={'sex_male':'gender'}) #TRUE - male,  False = female
# Wyświetlenie danych po zmianach
print(df)
print('\nSuma pustych pól:')
print (df.isna().sum()) #suma pustych pól
# W kolumnie 'age' gdzie jest brak wartości - przypisz średnią
for col in ['age']:
    mean_ = df[col].mean()    # liczymy średnią w kolumnie 'age'
    df[col].replace(np.NaN, mean_, inplace=True)   # wpisujemy średnią w miejsce pustych pól
# Wyświetlenie danych po zmianach
print(df)

# Regresja logistyczna
print('\nRegresja logistyczna\n')
X = df.iloc[:,1:]
print("Wyświetlenie X\n")
print(X)
y = df.survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Skuteczność
print('\nSkuteczność:')
print(model.score(X_test, y_test))
# Macierz konfuzji
print('\nMacierz konfuzji:')
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))


# Sprawdzenie, czy klasy są zablansowane
print ('\nSprawdzenie, czy klasy są zbalansowane')
print(df.gender.value_counts())

# Algorytm regresji logistycznej dla danych zbalansowanych
print('\nZmiana danych, zrównoważenie klas')
df1 = df.query('survived==0').sample(n=150)
df2 = df.query('survived==1').sample(n=150)
df3 = pd.concat([df1, df2])

X = df3.iloc[:,1:]
y = df3.survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)

# Skuteczność
print('\nKlasy zbalansowane - skuteczność algorytmu')
print(model.score(X_test, y_test))
# Macierz konfuzji
print('\nKlasy zbalansowane - macierz konfuzji')
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))