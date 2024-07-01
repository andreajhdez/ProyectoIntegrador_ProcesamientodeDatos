import pandas as pd
import matplotlib.pyplot as plt

# Leer el archivo CSV
df = pd.read_csv('registros_ventas_limpios_con_categoria_edad.csv')

# Contar los valores de cada categoría
anaemic_count = df['anaemia'].sum()
diabetic_count = df['diabetes'].sum()
smoker_count = df['smoking'].sum()
death_count = df['DEATH_EVENT'].sum()

# Etiquetas para las categorías
labels = ['Anémicos', 'Diabéticos', 'Fumadores', 'Muertos']

# Valores para las categorías
counts = [anaemic_count, diabetic_count, smoker_count, death_count]

# Crear la figura y los ejes de los subplots en una cuadrícula de 1x4
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

# Graficar las gráficas de torta para cada categoría
for ax, count, label in zip(axs, counts, labels):
    # Calcular porcentaje
    total = len(df)
    porcentaje = count / total * 100
    
    # Configurar etiquetas para la gráfica de torta
    labels_torta = ['Si', 'No']
    
    # Graficar la torta dentro del subplot actual
    ax.pie([count, total - count], labels=labels_torta, autopct='%1.1f%%', startangle=0)
    
    # Configurar título del subplot
    ax.set_title(label)
    
    
# Ajustar automáticamente el diseño de los subplots
plt.tight_layout()

# Mostrar la gráfica
plt.savefig('graficas_torta.png', dpi=300)
plt.show()
