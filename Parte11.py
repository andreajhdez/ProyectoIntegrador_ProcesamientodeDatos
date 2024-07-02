from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar los datos
file_name = 'registros_ventas_limpios_con_categoria_edad.csv'
df = pd.read_csv(file_name)

# Eliminar la columna 'categoria_edad'
df = df.drop(columns=['categoria_edad'])

# Visualizar la distribución de clases
plt.figure(figsize=(8, 6))
df['DEATH_EVENT'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Distribución de Clases')
plt.xlabel('DEATH_EVENT')
plt.ylabel('Número de Registros')
plt.xticks(ticks=[0, 1], labels=['No', 'Si'])
plt.savefig('distribucion_clases.png')  # Guardar la gráfica en formato PNG
plt.show()

# Separar las características y la variable objetivo
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# Dividir el conjunto de datos en entrenamiento y prueba de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

print("Distribución en el conjunto de entrenamiento:")
print(y_train.value_counts(normalize=True))
print("\nDistribución en el conjunto de prueba:")
print(y_test.value_counts(normalize=True))

# Ajustar un árbol de decisión
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión inicial: {accuracy:.2f}')


# Definir el rango de parámetros a probar
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

# Usar GridSearchCV para encontrar los mejores hiperparámetros
grid_search = GridSearchCV(DecisionTreeClassifier(
    random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros y entrenar el modelo con ellos
best_params = grid_search.best_params_
print(f'Mejores hiperparámetros: {best_params}')

clf_best = DecisionTreeClassifier(**best_params, random_state=42)
clf_best.fit(X_train, y_train)

# Predecir en el conjunto de prueba con el modelo optimizado
y_pred_best = clf_best.predict(X_test)

# Calcular la precisión con el modelo optimizado
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Precisión optimizada: {accuracy_best:.2f}')
