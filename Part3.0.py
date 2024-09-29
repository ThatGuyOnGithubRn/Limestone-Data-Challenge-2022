import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform

# Load data
stock_prices = pd.read_csv('data_challenge_stock_prices.csv')
index_prices = pd.read_csv('data_challenge_index_prices.csv')

# Calculate returns for stocks


stock_prices=stock_prices.to_numpy()
stock_return_prices=[0]*199999
i=0
while i<199999:
    stock_return_prices[i]=10000*((stock_prices[i+1]-stock_prices[i])/stock_prices[i])
    i+=1
# dfStock=pd.DataFrame(stock_return_prices)

index_prices=index_prices.to_numpy()
index_return_prices=[0]*199999
i=0
while i<199999:
    index_return_prices[i]=10000*((index_prices[i+1]-index_prices[i])/index_prices[i])
    i+=1
# print(index_return_prices)
# print(index_prices)
dfIndex=pd.DataFrame(index_return_prices)

result = np.hstack((stock_return_prices,index_return_prices))
result=pd.DataFrame(result)
corrCoeff=result.corr()
distanceFromOtherStocks = 1 - corrCoeff.to_numpy()


distanceFromOtherStocks2=distanceFromOtherStocks[:100,:100]
hierarchy = linkage(squareform(distanceFromOtherStocks2), method='average')
labels = fcluster(hierarchy, 0.9, criterion='distance')


plt.figure(figsize=(20, 15))
dn = dendrogram(hierarchy, color_threshold=0.9)
plt.title(f'Dendrogram')
plt.xlabel('Stocks')
plt.ylabel('Distance')
plt.show()


clusters=[]
for i in np.unique(labels):
    clusters.append(i)
arr=[]
for i in range(0,len(clusters)):
    arr.append([])
for i in range(1,101):
    for j in clusters:
        if labels[i-1]==j:
            arr[j-1].append(i)

for i in range(0,15):
    val=10000000000000
    clustno=0
    distanceOfIndex=distanceFromOtherStocks[100+i,:100]
    for clust in arr:
        clustno=clustno+1
        val2=0
        for item in clust:
            val2=distanceOfIndex[item-1]+val2
        # print(len(clust))
        val2=val2/len(clust)
        print(val,val2)
        if(val>val2):
            minClust=clustno
            print(minClust)
        val=min(val2,val)
    print(f"{i+1}: {minClust}")
