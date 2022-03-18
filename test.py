from datetime import date
import numpy as np
import pandas as pd
from pandas.core.indexes.base import Index
import xlsxwriter 
import pandas_datareader as web
import matplotlib.pyplot as plt


from datetime import date, datetime
import pandas as pd
import numpy as np
from pandas._libs.tslibs import Timestamp
import matplotlib.pyplot as plt

my_data = pd.read_csv("TradingData.csv")
my_data.drop(["Series","Symbol",  ], axis=1, inplace=True)
# print(my_data)
corr_matrix = my_data.corr()
v=corr_matrix['N.CLOSE'].sort_values(ascending=False)
print(v)



# my_data = my_data['TIMESTAMP OPEN HIGH       LOW     CLOSE      LAST  PREVCLOSE  TOTTRDQTY    TOTTRDVAL   P.OPEN P.CLOSE P.HIGH      P.LOW     P.CLOSE      P.LAST P.PREVCLOSE'.split()]

# Rename columns
# my_data.columns = 'DATE OPEN HIGH       LOW     CLOSE      LAST  PREVCLOSE  TOTTRDQTY    TOTTRDVAL   P.OPEN P.CLOSE P.HIGH      P.LOW     P.CLOSE      P.LAST P.PREVCLOSE'.split()


my_data['DATE'] = my_data['DATE'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))

my_data['DATE'] = my_data['DATE'].apply(lambda x: x.date())

my_data.set_index('DATE', inplace=True)



# train_test_split

from sklearn.model_selection import train_test_split
train_set, test_set  = train_test_split(my_data, test_size=0.2, random_state=42)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")




my_data= train_set.drop("N.CLOSE", axis=1)
my_data_labels = train_set["N.CLOSE"].copy()


# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
# imputer.fit(my_data)
my_data=my_data
my_data_labels=my_data_labels

# X = imputer.transform(my_data)
# food_tr = pd.DataFrame(X, columns=my_data.columns)


#pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    # ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

my_data_num_tr = my_pipeline.fit_transform(my_data)



#models
# from easyesn import PredictionESN
from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
#from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=2)
model = LinearRegression()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()


model.fit(my_data_num_tr, my_data_labels)

# some_data = my_data.iloc[:10]
# some_labels = my_data_labels.iloc[:10]
# prepared_data = my_pipeline.transform(some_data)
# Example =model.predict(prepared_data)
# print(Example)


from sklearn.metrics import mean_squared_error
my_data_predictions = model.predict(my_data_num_tr)
mse = mean_squared_error(my_data_labels, my_data_predictions)
rmse = np.sqrt(mse)
print("Root mean squared error is=",rmse)

new_data = pd.read_csv('2021TradingData.csv')
new_data.drop(['Symbol',	'Series'],axis=1, inplace=True)


new_data['DATE'] = new_data['DATE'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))

new_data['DATE'] = new_data['DATE'].apply(lambda x: x.date())

new_data.set_index('DATE', inplace=True)


# new_data = new_data.drop("x", axis=1)

Y_test = new_data["N.CLOSE"].copy()
new_data = new_data.drop("N.CLOSE", axis=1)


new_data_prepared = my_pipeline.transform(new_data)
final_predictions = model.predict(new_data_prepared)
# final_mse = mean_squared_error(Y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)
x=final_predictions-Y_test
print(x)
new_data['final_predictions']=final_predictions
new_data['labels']= Y_test
# some_labels=Y_test
# print("final_predictions are \n", list(some_predictions))
# print("value in LAST \n",list(some_labels))

# print("final RMSE :",final_rmse)
# for i in range len(new_data)

new_data[['final_predictions','labels' ]].plot(figsize = (10,4))
plt.grid(True)
plt.title('comparision' )
plt.axis('tight')
plt.ylabel("Price")
plt.show()

new_data.to_csv('predictionstest.csv')






#DATA
my_data = pd.read_csv('predictionstest.csv')


# def Buy_Sell_Indicator(my_data, i):
percentchange = []
position = None
num = 0

for i in range(len(my_data)):        
    prev_close = my_data.loc[i,'PREV_CLOSE']
    close = my_data.loc[i,'CLOSE']
    nextclose = my_data.loc[i,'final_predictions']

    if close<nextclose and position != 'buy' :
        position = 'buy'
        bp = close
        print(f"{position}: at {round(my_data['CLOSE'][i],2)} and Date {i}")

    if  nextclose<close and position != 'sell' and position == 'buy' :
        position = 'sell'
        sp = close
        print(f"{position}: at {round(my_data['CLOSE'][i],2)} and Date {i}")
        
    
               
        pc = (sp/bp-1)*100
        percentchange.append(pc)

    if (num== my_data['CLOSE'].count()-1 and position==1):
        position = 0
        sp = close
        pc = (sp/bp-1)*100
        percentchange.append(pc)
    num+= 1


gains = 0
ng = 0
losses = 0
nl = 0
totalR = 1
for i in percentchange:
    if (i>0):
        gains += i
        ng+= 1
    else : 
        losses +=i
        nl += 1
    totalR = totalR*((i/100)+1)
totalR = round ((totalR-1)*100,2)
if (ng >0) : 
    avggain = gains/ng
    MaxR = str(max(percentchange))
else : 
    avggain = 0 
    MaxR = "undefined"
if (nl>0):
    avgloss = losses/nl
    maxl = str(min(percentchange))
    ratio= str (-avggain/avgloss)
else :
    avgloss = 0
    maxl = "undefined"
    ratio = "inf"
if (ng>0 or nl > 0):
    battingavg = ng/(ng+nl)
else :
    battingavg = 0
print()
print("results for "+ "going back to  " + str (my_data.index[0])+ ",samplesize  : " +str (ng +nl)+" trades")
print("battingavg : " +str(battingavg))
print("gain/loss ratio :" +ratio)
print("average gain :" + str(avggain))
print("average loss:" +str(avgloss) )
print("maxreturns :" +MaxR)
print("Maxloss:" + maxl)
print("total return over " + str(ng+nl)+"trades : "+str(totalR) + "%")
print()