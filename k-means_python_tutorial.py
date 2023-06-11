from sklearn.cluster import KMeans
import warnings
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.DataFrame(columns=['x', 'y'])
df.loc[0] = [2,3]
df.loc[1] = [2, 18]
df.loc[2] = [4, 5]
df.loc[3] = [5, 3]
df.loc[4] = [7, 4]
df.loc[5] = [9, 19]
df.loc[6] = [10, 11]
df.loc[7] = [10, 5]
df.loc[8] = [6, 6]
df.loc[9] = [1, 1]

print(df.head(10))

sb.lmplot(x='x', y='y', data=df, fit_reg=False, scatter_kws={"s": 100})
plt.title('K-means Example')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

points = df.values
kmeans = KMeans(n_clusters = 2, n_init = 10).fit(points)
kmeans.cluster_centers_
warnings.filterwarnings("ignore", category=FutureWarning)
print(kmeans.cluster_centers_)
print(kmeans.labels_)

df['cluster'] = kmeans.labels_
print(df.head(10))

sb.lmplot(x='x', y='y', data=df, fit_reg=False, scatter_kws={"s": 150}, hue='cluster')
plt.title('K-means Example')
plt.show()