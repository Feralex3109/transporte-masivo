# Desarrollo de un modelo de aprendizaje no supervisado para el proyecto de transporte masivo

# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Cargar el Dataset
data = pd.read_csv("transport_data.csv")
print("\nPrimeras filas del dataset:")
print(data.head())

# Selección de características
features = data[["origin_lat", "origin_long", "destination_lat", "destination_long"]]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(features_scaled)
print("\nClusters asignados:")
print(data[['user_id', 'cluster']].head())

# Visualización de Clusters
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=data['cluster'], cmap='viridis')
plt.title("Clustering de Usuarios")
plt.xlabel("Latitud Origen (normalizada)")
plt.ylabel("Longitud Origen (normalizada)")
plt.colorbar(label="Cluster")
plt.show()

# Guardar Resultados
# Exportar los datos con los clusters asignados
data.to_csv("transport_data_with_clusters.csv", index=False)
print("Resultados guardados en transport_data_with_clusters.csv")










