# Importy bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Odczytanie pliku .csv z danymi
df = pd.read_csv('tips.csv')
# Wyświetlenie typu danych
print(type(df))
#Sprawdzenie danych
print('\nSprawdzenie danych:')
print(df.describe().T.to_string())
# Zliczenie ilości kobiet i mężczyzn w pliku z danymi
print (f'\nZliczanie wartosci kolumny sex\n{df.sex.value_counts()}')
# Usunięcie kolumn
del (df['time'])
del (df['smoker'])
del (df['day'])
del (df['size'])
# Wyświetlenie danych po usunięciu kolumn
print(df)
# Zamiana wartości na dane numeryczne
df = pd.get_dummies(df)
# Usunięcie nadmiarowej kolumny
del (df['sex_Female'])
# Zmiana nazwy kolumny, wartości True = Male, wartości False = Female
df = df.rename(columns={'sex_Male':'gender'})
# Wyświetlenie danych po zmianach
print(df)

# Regresja liniowa - algorytm
model = LinearRegression()
model.fit(df[['gender','total_bill']],df['tip'])
print('\nWyniki regresji liniowej:\n')
print(f'Współczynnik kierunkowy: {model.coef_}\nWyraz wolny: {model.intercept_}')
# Wyświetlenie równania
print('\nRównanie:')
print(f'tip = gender * {model.coef_[0]} + total_bill * {model.coef_[1]} + {model.intercept_}')

#Sprawdzenie modelu
print('\nSprawdzenie modelu Liniowej regresji, przykład dla mężczyzny rachunek 20, kwota napiwku =')
print(model.predict([[1, 20]]))

# Korelacja
print('\nKorelacja:')
print(df.corr())
sns.heatmap(df.corr(), annot=True)
plt.show()

# Ponowne użycie algorytmu Regresji liniowej, tym razem z podziałem na dane testowe i uczące
X = df.iloc[:,[0,2]] # Kolumny 'total_bill' oraz 'gender'
y = df.tip
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
# Skuteczność
print('\nSkuteczność modelu wynosi:')
print(model.score(X_test, y_test))