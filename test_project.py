# -*- coding: utf-8 -*-
"""Project Machine Learning - Online Retail.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Co1vPBtIjHFp001t85JjnWGCeHFQxxo8
"""

import numpy as np
import pandas as pd
import os
import seaborn as sns
import datetime as dt
import plotly.express as px
import sklearn
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.mixture import GaussianMixture as GMM
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

# read file
path = 'online_retail_II.csv'
data = pd.read_csv(path)
print("Data Describe ===========================")
print(data.describe())
print("=========================================")

data.isnull().sum()

data.info()

# ngatur maksimal data yang ditampilkan
pd.set_option("display.max_rows", 101)
pd.set_option("display.max_columns", 10)

# nyimpen data yang price sama quantity nya < 0 utk dihapus
negprice = data[data['Price'] < 0].sum()
negquantity = data[data['Quantity'] < 0].count()
print("Negprice =========================================")
print(negprice)
print("Negquantity =========================================")
print(negquantity)
print("=========================================")


np.set_printoptions(threshold=np.inf)
data['Description'].unique()

# ngedrop nilai kosong
df_new = data['Description']
df_new = df_new.dropna()
df_new.dropna()
df_new = pd.DataFrame(df_new)


my_list = [y for x in df_new['Description'] for y in x.split() if y.islower()]
mylist = list(set(my_list))
print("mylist =========================================")
print(mylist)
print("=========================================")

##menghapus data yang tidak sesuai (beli barang tapi nama barangnya gaada)
data_bad = data[data['Description'].isin([ mylist ])]

data = data[~data.apply(tuple,1).isin(data_bad.apply(tuple,1))]

##mengubah data Customer ID null menjadi 99999
data[['Customer ID']] =data[['Customer ID']].fillna(99999)
#replace null description values with 'Unknown'
data[['Description']] =data[['Description']].fillna('Unknown')

##menghapus data dengan pembelian negatif dan ID customer yang tidak sesuai
# nyisain nilai yang di atas 0 dan customer id yang sesungguhnya
data = data[data['Quantity'] > 0]
data = data[data['Customer ID'] != 99999]
##menghapus data pengembalian barang
data = data[~data["Invoice"].str.contains("C", na = False)]

data.info()

data = data.drop_duplicates()
data.shape

data.info()

#data.sort_values(by=['Quantity'])
data_uk = data[data.Country == 'United Kingdom']
data_uk.head(5)

#https://datascience.stackexchange.com/questions/64260/pearson-vs-spearman-vs-kendall/64261
correlation=data_uk.corr(method='kendall')

print("CORRELATION =========================================")
print(correlation)
print("=========================================")

heatmap = sns.heatmap(correlation, vmin=-1, vmax=1, annot=True)

# heatmap = sns.boxplot(x='Quantity', data=data_uk)
#
# heatmap = sns.boxplot(x='Price', data=data_uk)
plt.show()

# Recency Frequency Monetary (RFM)
# cari total nilai setiap transakti
amount = pd.DataFrame(data_uk.Quantity * data.Price, columns=['Amount'])

# menggabungkan customer id dan amount
data_cust = np.array(data_uk['Customer ID'], dtype=np.object)
data_cust = pd.DataFrame(data_cust, columns=["Customer ID"])
data_cust = pd.concat(objs=[data_cust, amount], axis=1, ignore_index=False)

# cari nilai monetary
monetary = data_cust.groupby(by=["Customer ID"]).Amount.sum()
monetary = monetary.reset_index()
monetary = monetary[monetary['Customer ID'] != 99999]

print("MONETARY =========================================")
print(monetary)
print("=========================================")

# cari nilai frequency
frequency = data_uk[['Customer ID', 'Invoice']]

frequency_df = frequency.groupby("Customer ID").Invoice.count()
frequency_df = pd.DataFrame(frequency_df)
frequency_df = frequency_df.reset_index()
frequency_df.columns = ["Customer ID", "Frequency"]

print("FREQUENCY =========================================")
print(frequency_df)
print("=========================================")

date = data_uk["InvoiceDate"].max()  # Last invoice date
# type(date)
date_split = date.split(' ')  # DATENYA TYPE STRING
date_split = date_split[0].split('-')
date_split = list(map(int, date_split))

# dateStr = date.strftime("%Y-%m-%d") // IN CASE DATE-NYA TYPE DATE
# dt.strptime(date, '%Y-%m-%d %H:%M:%S.%f').strftime('%m/%d/%Y') // IN CASE DATE-NYA TYPE OBJECT

print("DATE SPLIT =========================================")
print(date_split)
print("=========================================")

# cari nilai recency

today_date = dt.datetime(date_split[0], date_split[1],
                         date_split[2])  # last invoice date is assigned to today_date variable
data_uk["Customer ID"] = data_uk["Customer ID"].astype(int)
data_uk["InvoiceDate"] = pd.to_datetime(data_uk["InvoiceDate"])

# Grouping the last invoice dates according to the Customer ID variable, subtracting them from today_date, and assigning them as recency
recency = (today_date - data_uk.groupby("Customer ID").agg({"InvoiceDate": "max"}))
# Rename column name as Recency
recency.rename(columns={"InvoiceDate": "Recency"}, inplace=True)
# Change the values to day format
recency_df = recency["Recency"].apply(lambda x: x.days)

print("RECENCY =========================================")
print(recency_df)
print("=========================================")

# penggabungan data frame
RFM = frequency_df.merge(monetary, on="Customer ID")
RFM = RFM.merge(recency_df, on="Customer ID")
print("RFM =========================================")
print(RFM)
print("=========================================")

fig = px.box(RFM, y="Amount")
fig.show()

fig = px.box(RFM, y="Frequency")
fig.show()

# outlier treatment: We will delete everything outside the IQR

Q1 = RFM.Amount.quantile(0.25)
Q3 = RFM.Amount.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Amount >= (Q1 - 1.5 * IQR)) & (RFM.Amount <= (Q1 + 1.5 * IQR))]

# outlier treatment : We will delete everything outside the IQR

Q1 = RFM.Frequency.quantile(0.25)
Q3 = RFM.Frequency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Frequency >= (Q1 - 1.5 * IQR)) & (RFM.Frequency <= (Q1 + 1.5 * IQR))]

fig = px.box(RFM, y="Recency")
fig.show()

# menghapus customer id dari rfm buat di standarisasi

RFM = RFM.drop(['Customer ID'], axis=1)
standard_scaler = StandardScaler()
RFM = standard_scaler.fit_transform(RFM)

# RFM = pd.DataFrame(RFM)
RFM = pd.DataFrame(data=RFM, columns=['Frequency', 'Amount', 'Recency'])

print("RFM =========================================")
print(RFM)
print("=========================================")

# Expectation-Maximization (EM) using Gaussian Mixture Model (GMM)

# At its simplest, GMM is also a type of clustering algorithm. As its name implies, each cluster is modelled according to a different Gaussian distribution.
# This flexible and probabilistic approach to modelling the data means that rather than having hard assignments into clusters like k-means, we have soft assignments.
# This means that each data point could have been generated by any of the distributions with a corresponding probability.

# Let's calculate the silhouette score for various number of n_components
for n_components in range(2, 15):
    clusterer = GMM(n_components=n_components)
    preds = clusterer.fit_predict(RFM)

    score = sklearn.metrics.silhouette_score(RFM, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_components, score))



print(__doc__)

lowest_bic = np.infty
bic = []
n_components_range = range(1, 9)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = GMM(n_components=n_components,
                  covariance_type=cv_type)
        gmm.fit(RFM)
        bic.append(gmm.bic(RFM))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(20, 10))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
       .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)
plt.show()

n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(RFM)
          for n in n_components]

plt.plot(n_components, [m.bic(RFM) for m in models], label='BIC')
plt.plot(n_components, [m.aic(RFM) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components')

RFM_gmm = RFM.copy()

RFM_gmm['gmm'] = GMM(n_components=5, random_state=42).fit_predict(RFM)

fig = px.scatter_3d(RFM_gmm, x='Frequency', y='Amount', z='Recency',
                    color='gmm')
fig.show()

# Dari hasil percobaan antara 3-4-5, dihasilkan bahwa angka clustering yang paling optimal adalah 5 cluster
