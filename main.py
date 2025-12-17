import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('house_rent.csv')
#print(df.head())
#print(df.info())
#print(df.describe())

mean_age = df['age'].mean()

df['age'] = df['age'].fillna(mean_age)

df.drop_duplicates(inplace=True)

X = df[['size_sqft', 'bedrooms', 'age']]
y = df['rent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predicted rents for the test set:")
print(y_pred)



output
C:\Users\Student\PycharmProjects\HouseRentPrediction\.venv\Scripts\python.exe C:\Users\Student\PycharmProjects\HouseRentPrediction\main.py 
Predicted rents for the test set:
[11426.0063677  20326.38151887]

Process finished with exit code 0
