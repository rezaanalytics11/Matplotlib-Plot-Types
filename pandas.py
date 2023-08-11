import pandas
from sklearn.linear_model import LinearRegression

df = pandas.read_csv(r"C:\Users\Ariya Rayaneh\Desktop\data0.csv")

print(df,5*'\n')

X=df[['Volume','Weight']]
Y=df.CO2
model=LinearRegression()
model.fit(X,Y)

y_predict=model.predict([[800,500]])

print(df.head(5))
print(y_predict)



