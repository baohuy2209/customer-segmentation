import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Tạo dữ liệu mẫu: 300 điểm, 4 cụm
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Phương pháp Elbow để chọn k
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Phương pháp Elbow để chọn k')
plt.xlabel('Số cụm (k)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.savefig('elbow_plot.png')
plt.close()

# Chọn k = 4 dựa trên Elbow (giả định điểm khuỷu)
k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Trực quan hóa kết quả phân cụm
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', label='Dữ liệu')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Tâm cụm')
plt.title(f'K-means với k={k}')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_result.png')
plt.close()