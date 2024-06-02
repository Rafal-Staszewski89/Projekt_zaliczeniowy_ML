# Importy bibliotek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Zaczytanie danych
df = pd.read_csv('penguins.csv')
print(df)
print('\nSuma pustych pól w poszczególnych komórkach')
print (df.isna().sum()) #suma pustych pól
# Zastąpienie pustych rekordów średnią z danej kolumny
for col in ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']:
    mean_ = df[col].mean()    # liczymy średnią w danej kolumnie
    df[col].replace(np.NaN, mean_, inplace=True)   # wpisujemy srednią w miejsce pustych pól
# Zastąpienie pustych pól w kolumnie 'sex' wartością 'MALE'
for col in ['sex']:
    df[col].replace(np.NaN, 'MALE', inplace=True)
print('\nSuma pustych pól po modyfikacji - sprawdzenie')
print (df.isna().sum()) #suma pustych pól, sprawdzenie

print('\nSprawdzenie, czy klasy są zabalnsowane')
print(df.species.value_counts()) #Czy klasy są zabalansowane?

# Zastąpienie danych wartościami numerycznymi
isla = {
    'Torgersen': 0, 'Biscoe': 1, 'Dream': 2,
}
df['island_value'] = df['island'].map(isla)

sxs = {
    'MALE': True, "FEMALE": False
}
df['sex_value'] = df['sex'].map(sxs)

spec = {
    'Adelie': 0, 'Gentoo': 1, 'Chinstrap': 2
}
df['species_value'] = df['species'].map(spec)

# Usunięcie kolumn
del (df['sex'])
del (df['species'])
del (df['island'])

# Sprawdzenie danych
print(df.describe().to_string())

# Wykres zależności pomiędzy długością a szerekością dzioba pingwina, w odniesieniu do gatunku
sns.scatterplot(data=df, x='bill_length_mm', y='bill_depth_mm', hue='species_value')
plt.show()

print('\nKlasyfikator KNN\n')

X = df.iloc[:,[0,1]] # Dwie pierwsze kolumny, wszystkie wiersze
print("Wyświetl X\n")
print(X)
y = df.species_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier() #(n_neighbors=4, weights='uniform')
model.fit(X_train, y_train)
#Skuteczność
print('\nSkuteczność algorytmu:')
print(model.score(X_test, y_test))
print('\nConfusion matrix:')
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))