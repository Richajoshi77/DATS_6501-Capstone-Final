# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Importing Data Dictionary

xls = pd.ExcelFile('Data_Dictionary.xlsx')
Data_Dictionary1 = pd.read_excel(xls, 'train')
Data_Dictionary2 = pd.read_excel(xls, 'merchant')
Data_Dictionary2 = pd.read_excel(xls, 'history')
Data_Dictionary2 = pd.read_excel(xls, 'new_merchant_period')


# As we will be training the "train.csv" file and testing it on the "test.csv" file, so we are reading these 
# files using the pandas function "read_csv". "first_active_month" column in both the files is used knowing the 
# first time the card got activated by respective customers. This column is parsed as date and time column.

train_df = pd.read_csv("train.csv", parse_dates=["first_active_month"]) 
test_df = pd.read_csv("test.csv", parse_dates=["first_active_month"])


# Knowing the size of the dataset.


print("Number of rows and columns in train set : ",train_df.shape)
print("Number of rows and columns in test set : ",test_df.shape)


%matplotlib inline

plt.figure(figsize=[18,6])
plt.suptitle('Feature distributions in train data and test data', fontsize=20, y=1.1)
for num, col in enumerate(['feature_1', 'feature_2', 'feature_3']):
    plt.subplot(2, 4, num+1)
    if col is not 'target':
        v_c = train_df[col].value_counts() / train_df.shape[0]
        plt.bar(v_c.index, v_c, label=('train'), align='edge', width=-0.3, edgecolor=[0.2]*3)
        v_c = test_df[col].value_counts() / test_df.shape[0]
        plt.bar(v_c.index, v_c, label=('test'), align='edge', width=0.3, edgecolor=[0.2]*3)
        plt.title(col)
        plt.legend()
plt.show()



# Information of both the dataframes

print("Basic information of Train Dataframe: ")
print(train_df.info())

print(" ")
print("Basic information of Test Dataframe: ")
print(test_df.info())


# Printing the data types of data frames

print("Data types of Train set: ")
print(train_df.dtypes)

print( " ")
print("Data types of Test set: ")
print(test_df.dtypes)


# Printing the head of train data frame.

print("Head of Train set:")
train_df.head()


# Printing the head of train data frame.

print("Head of Test set:")
test_df.head()


# It represents the statistical information like the count, mean, std, min, 25%, 50%, 75% and maximum observation in the data.

print("Statistical information of Train Dataframe: ")
print(train_df.describe())  

# Printing the statistical information of test dataset.

print("Statistical information of Test Dataframe: ")
print(test_df.describe())


# Plotting the "target" column from train_df


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df["target"].values),color = 'maroon')
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.show()


# Histogram of loyalty score

# As from the above plot, not a good conclusion of the loyalty score can be seen therefore we go for the histogram plot. 
# It can be seen that it follows a normal distribution curve for the loyalty score with a mean of zero and standard 
# deviation of +- 4sigma. For the data points, which are out of 4sigma that data points are the outliers in this case values 
# greater than -30 are consider as outlier values.

target_col = "target"   # variable for target_col

plt.figure(figsize=(8,6))
sns.distplot(train_df[target_col].values, bins=50, color="brown")
plt.title("Histogram of Loyalty score")
plt.xlabel('Loyalty score or Target Column', fontsize=14)
plt.ylabel('Relative frequency of scores', fontsize=14)
plt.show()


# Counting the numbers of id which is below score -30. There are total 2207 of them.
(train_df['target']<-30).sum()

# Making plots of other features of dataset

# We are plotting the Violin Plots for the features. These are similar to box plots, except that it shows 
# the kernel probability density of data at different points. So, this violin plot is plotted for all the features 
# with the loyalty scores.

colors = [
                    '#A040A0',  
                    '#F8D030',  
                    '#EE99AC',  
                    '#C03028',  
                    '#F85888',  
                    '#B8A038',  
                    '#705898',  
                    '#98D8D8',  
                    '#7038F8',  
                   ]
# feature 1
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_1", palette=colors, y= target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 1 distribution")
plt.show()


# feature 2
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_2", palette='Blues', y= target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 2', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 2 distribution")
plt.show()

# feature 3
plt.figure(figsize=(8,4))
sns.violinplot(x="feature_3", y= target_col, data=train_df)
plt.xticks(rotation='vertical')
plt.xlabel('Feature 3', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Feature 3 distribution")
plt.show()

print("Plots gives the density destribution of each category of each feature.")



# Bar plots of all the categories of all the feature columns

fig, ax = plt.subplots(1, 3, figsize = (16, 6))
train_df['feature_1'].value_counts().sort_index().plot(kind='barh', align='center',ax=ax[0], color='darkgreen', title='feature_1')
train_df['feature_2'].value_counts().sort_index().plot(kind='barh', ax=ax[1], color='0.62', title='feature_2')
train_df['feature_3'].value_counts().sort_index().plot(kind='barh', ax=ax[2], color='r', title='feature_3')
plt.suptitle('Counts of categiories for features')

# checking missing data for train_df. There are no missing values in train data set.

total = train_df.isnull().sum().sort_values(ascending = False)
percent = (train_df.isnull().sum()/train_df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# checking missing data for test_df

total = test_df.isnull().sum().sort_values(ascending = False)
percent = (test_df.isnull().sum()/test_df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

d1 = train_df['first_active_month'].value_counts().sort_index()
d2 = test_df['first_active_month'].value_counts().sort_index()
data = [go.Scatter(x=d1.index, y=d1.values, name='train'), go.Scatter(x=d2.index, y=d2.values, name='test')]
layout = go.Layout(dict(title = "Counts of first active",xaxis = dict(title = 'Month'), yaxis = dict(title = 'Count'),
                  ),legend=dict(orientation="v"))
py.iplot(dict(data=data, layout=layout))


# Created a elapsed time column for each id in train set:

import datetime
def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df
train_df = read_data('train.csv')
test_df = read_data('test.csv')

target = train_df['target']


# Forming the corelation Matrix of train_df

train_df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(train_df.corr(),cmap=cmap, vmax=.3, square=True, linewidths=.5, cbar_kws={"shrink": .5})


from pandas.plotting import scatter_matrix
select_cols = ['feature_1', 'feature_2', 'feature_3', 'target']
scatter_matrix(train_df[select_cols], figsize=[10,10], color = 'red')
plt.suptitle('Pair-wise scatter plots for columns in train', fontsize=15)
plt.show()


# Deleting the target column

del train_df['target']

# Printing the head of the dataset

train_df.head()


# Import old Merchant data

merchant_df = pd.read_csv("merchants.csv")
print("shape of merchant : ",merchant_df.shape)
merchant_df.head()


# checking missing data for old merchant dataset

total = merchant_df.isnull().sum().sort_values(ascending = False)
percent = (merchant_df.isnull().sum()/merchant_df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


Mer = merchant_df.loc[(merchant_df['numerical_1'] < 0.1) &
                               (merchant_df['numerical_2'] < 0.1) &
                               (merchant_df['avg_sales_lag3'] < 5) &
                               (merchant_df['avg_purchases_lag3'] < 5) &
                               (merchant_df['avg_sales_lag6'] < 10) &
                               (merchant_df['avg_purchases_lag6'] < 10) &
                               (merchant_df['avg_sales_lag12'] < 10) &
                               (merchant_df['avg_purchases_lag12'] < 10)]


cat_cols = ['active_months_lag6','active_months_lag3','active_months_lag12','category_4']
num_cols = ['numerical_1', 'numerical_2','merchant_group_id','merchant_category_id','avg_sales_lag3', 'avg_purchases_lag3', 'avg_sales_lag6', 'avg_purchases_lag6', 'avg_sales_lag12', 'avg_purchases_lag12']

plt.figure(figsize=[15, 15])
plt.suptitle('Merchants table histograms', y=1.02, fontsize=20)
ncols = 4
nrows = int(np.ceil((len(cat_cols) + len(num_cols))/4))
last_ind = 0
for col in sorted(list(Mer.columns)):
    #print('processing column ' + col)
    if col in cat_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        vc = Mer[col].value_counts()
        x = np.array(vc.index)
        y = vc.values
        inds = np.argsort(x)
        x = x[inds].astype(str)
        y = y[inds]
        plt.bar(x, y, color=('lightblue'))
        plt.title(col, fontsize=15)
    if col in num_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        Mer[col].hist(bins = 50, color=('lightblue'))
        plt.title(col, fontsize=15)
    plt.tight_layout()


    # Values count for category 1 in old merchant data

temp = merchant_df["category_1"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_1 in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# Values count for category 2 in old merchant data

temp = merchant_df["category_2"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('category_2 in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# Values of most recent purchase category

temp = merchant_df["most_recent_purchases_range"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('most_recent_purchases_range in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


# Values of most recent sales category

temp = merchant_df["most_recent_sales_range"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (6,6))
plt.title('most_recent_sales_range in merchant')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()


corrs = np.abs(Mer.corr())
ordered_cols = (corrs).sum().sort_values().index
np.fill_diagonal(corrs.values, 0)
plt.figure(figsize=[10,10])
plt.imshow(corrs.loc[ordered_cols, ordered_cols], cmap='plasma', vmin=0, vmax=1)
plt.colorbar(shrink=0.7)
plt.xticks(range(corrs.shape[0]), list(ordered_cols), rotation=90)
plt.yticks(range(corrs.shape[0]), list(ordered_cols))
plt.title('Heat map of coefficients of correlation between merchant\'s features', fontsize=17)
plt.show()



# Import new merchant data file

new_merchant_df = pd.read_csv("new_merchant_transactions.csv")
print("shape of new_merchant_transactions : ",new_merchant_df.shape)


# checking missing data for old merchant dataset

total = new_merchant_df.isnull().sum().sort_values(ascending = False)
percent = (new_merchant_df.isnull().sum()/new_merchant_df.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# Check out the column authorized flag.
new_merchant_df['authorized_flag'].unique()  


categorical_columns = ['city_id', 'category_1', 'category_3','category_2', 'merchant_category_id', 'state_id', 'subsector_id']
new_merchant_df[categorical_columns].head()


# Now lets look at the historical data

hist_df = pd.read_csv("historical_transactions.csv")
hist_df.head()
hist_df.shape
hist_df.isnull().sum()


# convert the authorized_flag to a binary value

hist_df['authorized_flag'] = hist_df['authorized_flag'].map({'Y':1, 'N':0})


def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).\
                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(hist_df)


#  Distribution plot for month lag 

f, ax = plt.subplots(figsize=(14, 6))
sns.distplot(hist_df['month_lag'], color = 'green')


temp = history["hist_city_id_nunique"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
plt.figure(figsize = (18,8))
plt.title('hist_city_id_nunique')
sns.set_color_codes("pastel")
sns.barplot(x = 'labels', y="values", data=df)
locs, labels = plt.xticks()
plt.show()








