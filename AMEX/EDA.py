## **Import Libraries**

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats

## **Reading and Understanding The Provided Data**

amex = pd.read_csv("H:/Kaggle/AmericanExpress/train_data.csv", sep = ',')
print("The DataFrame consists of {} rows and {} features".format(amex.shape[0],amex.shape[1]))
amex.info(max_cols=190, show_counts=True)

# Count null values
var = amex.isnull().sum()
print(var.to_string())

# Separate quantitative and qualitative values
quant_amex_vars = amex.columns[amex.dtypes != object]
qual_amex_vars = amex.columns[amex.dtypes == object]

# Show the list of quantitative values
print(quant_amex_vars)

# Show the list of qualitative values
print(qual_amex_vars)

# Summary statistics on quantitative columns
amex[quant_amex_vars].describe()

# Summary statistics on qualitative columns
amex[qual_amex_vars].describe()

amex_target = pd.read_csv("H:/Kaggle/AmericanExpress/train_labels.csv", sep = ',')
amex_target.head(10)
amex_target.info(max_cols=190, show_counts=True)

# Count null values
var = amex_target.isnull().sum()
print(var.to_string())

# Summary statistics
amex_target.describe()

target=amex_target.target.value_counts(normalize=True)
target.rename(index={1:'Default',0:'Paid'},inplace=True)
colors = ['#17becf', '#E1396C']
data = plt.pie(target,
                labels= target.index,
                colors=colors,
                autopct='%1.1f%%'
                )
layout = plt.title("Target Distribution",fontdict = {'fontsize' : 20})
fig = plt.figure(figsize = (0.46,0.95))

# Histograms of each variable : 0.2 random sample of rows selected
# ---------------------------------------------------------------------
# random_state helps assure that you always get the same output when you split the data
# this helps create reproducible results and it does not actually matter what the number is
# frac is percentage of the data that will be returned
data_part = amex.sample(frac=0.6,random_state=42)
print(data_part.shape)

# plot the histogram of each parameter
data_part.hist(figsize=(25,25))
plt.show()

for col in (quant_amex_vars):
    print("Skewness of {} for ".format(amex[col].skew()),col)
    
## **Handle Duplicates**

amex_no_duplicates = amex.drop_duplicates(subset=['customer_ID'])
amex_no_duplicates.info(max_cols=190, show_counts=True)

# Count null values
var = amex_no_duplicates.isnull().sum()
print(var.to_string())

# Summary statistics on quantitative columns
amex_no_duplicates[quant_amex_vars].describe()

for col in quant_amex_vars:
    if amex_no_duplicates[col].isnull().sum() != 0:
        mean_value = amex_no_duplicates[col].mean(axis=0)
        amex_no_duplicates[col].replace(np.nan, mean_value, inplace=True)
        
# Summary statistics on qualitative columns
amex_no_duplicates[qual_amex_vars].describe()

## **Correlations**

amex_correlations = [ ]
top_features = [ ]
count = 0
for col in quant_amex_vars:
    pearson_coef, p_value = stats.pearsonr(amex_no_duplicates[col], amex_target['target'])
    amex_correlations.append(pearson_coef)
    if np.abs(pearson_coef) > 0.3:
        top_features.append(col)
        print(top_features[count], pearson_coef)
        count += 1
        
# Correlation Matrix
correlation_matrix = amex_no_duplicates[top_features].corr()
fig = plt.figure(figsize = (27,18))
sns.heatmap(correlation_matrix, vmax = .8, square = True)
plt.show()

## **Handle Missing Values**

# Count null values
var = amex_no_duplicates.isnull().sum()
print(var.to_string())

amex_no_duplicates.drop(['D_64'],axis=1,inplace=True)

## **Feature Exploration**
# Determine the number of Paid v Default cases
default = amex_target[amex_target['target'] == 1]
paid = amex_target[amex_target['target'] == 0] 

outlier_fraction = len(default) / float(len(paid))

print('Outlier_Fraction: {}'.format(outlier_fraction))
print('Default Cases: {}'.format(len(default)))
print('Paid Cases: {}'.format(len(paid)))
