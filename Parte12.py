import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

# Cargar los datos
file_name = 'registros_ventas_limpios_con_categoria_edad.csv'
df = pd.read_csv(file_name)

# Eliminar la columna 'categoria_edad'
df = df.drop(columns=['categoria_edad'])

# Separar las características y la variable objetivo
X = df.drop(columns=['DEATH_EVENT'])
y = df['DEATH_EVENT']

# Dividir el conjunto de datos en entrenamiento y prueba de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ajustar un Random Forest
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred_rf = rf_clf.predict(X_test)

# Calcular la precisión
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Calcular la matriz de confusión
cm = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm, display_labels=['No hubo muertes', 'Hubo muertes']).plot(cmap='Blues')
plt.title('Matriz de Confusión')
plt.savefig('matriz_confusion_random_forest.png')  # Guardar la gráfica en formato PNG
plt.show()

# Calcular el F1-Score
f1_rf = f1_score(y_test, y_pred_rf)

# Imprimir precisión y F1-Score
print(f'Precisión de Random Forest: {accuracy_rf:.2f}')
print(f'F1-Score de Random Forest: {f1_rf:.2f}')

# ¿El accuracy captura completamente el rendimiento del modelo?
if f1_rf < accuracy_rf:
    print("El F1-Score es menor que el accuracy, lo que sugiere que el modelo puede estar desequilibrado en cuanto a las clases.")
else:
    print("El F1-Score y el accuracy son comparables, lo que sugiere un buen equilibrio entre precisión y exhaustividad.")

# Optimización de hiperparámetros del Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Usar GridSearchCV para encontrar los mejores hiperparámetros
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='f1')
grid_search_rf.fit(X_train, y_train)

# Obtener los mejores parámetros y entrenar el modelo con ellos
best_params_rf = grid_search_rf.best_params_
print(f'Mejores hiperparámetros para Random Forest: {best_params_rf}')

rf_best = RandomForestClassifier(**best_params_rf, random_state=42)
rf_best.fit(X_train, y_train)

# Predecir en el conjunto de prueba con el modelo optimizado
y_pred_rf_best = rf_best.predict(X_test)

# Calcular la precisión y el F1-Score con el modelo optimizado
accuracy_rf_best = accuracy_score(y_test, y_pred_rf_best)
f1_rf_best = f1_score(y_test, y_pred_rf_best)
print(f'Precisión optimizada de Random Forest: {accuracy_rf_best:.2f}')
print(f'F1-Score optimizado de Random Forest: {f1_rf_best:.2f}')
