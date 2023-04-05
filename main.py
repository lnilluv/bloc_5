import pandas as pd

df = pd.read_csv('data/get_around_pricing_project.csv')

print(df.head())
print(df.info())
# Separate target variable Y from features X  
target_name = 'rental_price_per_day'  

print("Separating labels from features...")  

Y = df.loc[:,target_name]  
X = df.drop(target_name, axis = 1) 
print("xxxxxxxxxxxxxxxxxxx")
print(X.head())
# Automatically detect names of numeric/categorical columns  
numeric_features = []  
categorical_features = []  
for i,t in X.dtypes.iteritems():  
    if ('float' in str(t)) or ('int' in str(t)) :  
        numeric_features.append(i)  
    else :  
        categorical_features.append(i)  
  
print('Found numeric features ', numeric_features)  
print('Found categorical features ', categorical_features)