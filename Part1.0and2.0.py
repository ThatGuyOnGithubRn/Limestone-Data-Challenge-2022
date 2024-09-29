
# import numpy as np
# import pandas as pd
# # import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
# from scipy.spatial.distance import squareform
# from scipy import stats
# df = pd.read_csv("data_challenge_stock_prices.csv")
# stock_prices=df.to_numpy()
# return_prices=[0]*199999
# i=0
# while i<199999:
#     return_prices[i]=100*((stock_prices[i+1]-stock_prices[i])/stock_prices[i])
#     i+=1
# df=pd.DataFrame(return_prices)
# dff=df.corr()
# print(dff)
# csv_correlation_coefficient_array=dff.to_numpy()
# dissimilarity = 1 - csv_correlation_coefficient_array
# hierarchy = linkage(squareform(dissimilarity), method='average')
# labels = fcluster(hierarchy, 0.9, criterion='distance')
# fig = plt.figure(figsize=(25, 10))
# dn = dendrogram(hierarchy, color_threshold=0.9)
# plt.show()
# clusters = dict()
# for i in range(100):
#     if labels[i] not in clusters:
#         clusters[labels[i]] = [i]
#     else:
#         clusters[labels[i]].append(i)


# for i in range(0, len(clusters)):
#     print(i+1, clusters[i])







# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# Load stock prices data
stockdf = pd.read_csv("data_challenge_stock_prices.csv")
stock_prices = stockdf.to_numpy()

# Calculate stock returns
return_prices=[0]*199999
i=0
while i<199999:
    return_prices[i]=100*((stock_prices[i+1]-stock_prices[i])/stock_prices[i])
    i+=1
# stock_return_prices = 100 * (np.diff(stock_prices, axis=0) / stock_prices[:-1])
# stock_return_prices = np.vstack(([0]*stock_prices.shape[1], stock_return_prices))  # Add zero for the first row
stockdf = pd.DataFrame(return_prices)

# Calculate the correlation matrix
df = stockdf.corr()

# Convert correlation to distance
distanceFromOtherStocks = 1 - df.to_numpy()

# Perform hierarchical clustering
hierarchy = linkage(squareform(distanceFromOtherStocks), method='average')


labels = fcluster(hierarchy, 0.9, criterion='distance')
    
    # Create a new figure for each threshold
plt.figure(figsize=(20, 15))
dn = dendrogram(hierarchy, color_threshold=0.9)
plt.title(f'Dendrogram')
plt.xlabel('Stocks')
plt.ylabel('Distance')
plt.show()
    
    # Print the cluster labels for the current threshold
# print(f"Clusters: {np.unique(labels)}")
clusters=[]
for i in np.unique(labels):
    clusters.append(i)

print(clusters)

arr=[]
for i in range(0,len(clusters)):
    arr.append([])
for i in range(1,101):
    for j in clusters:
        if labels[i-1]==j:
            arr[j-1].append(i)


print(arr)






# import pandas as pd
# # from sklearn.neural_network import MLPRegressor
# import numpy as np
# # import pickle
# import matplotlib.pyplot as plt;
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram;
# from scipy.spatial.distance import squareform
# from scipy import stats


# stockdf = pd.read_csv("data_challenge_stock_prices.csv")
# stock_prices=stockdf.to_numpy()
# stock_return_prices=[0]*199999
# i=0
# while i<199999:
#     stock_return_prices[i]=100*((stock_prices[i+1]-stock_prices[i])/stock_prices[i])
#     i+=1
# stockdf=pd.DataFrame(stock_return_prices)
# df=stockdf.corr()

# distanceFromOtherStocks=df.to_numpy()
# distanceFromOtherStocks = 1 - distanceFromOtherStocks
# # print(squareform(distanceFromOtherStocks))
# hierarchy = linkage(squareform(distanceFromOtherStocks), method='average')
# thresholds=[0.001,0.2,0.9]
# for i in thresholds:
#     labels = fcluster(hierarchy, i, criterion='distance')
#     fig = plt.figure(figsize=(20, 15))
#     dn = dendrogram(hierarchy, color_threshold=0.9)
#     plt.show()