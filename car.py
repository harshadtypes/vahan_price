import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
data = pd.read_csv('vahan_data.csv')

# Display basic info
data_overview = data.info()
missing_values = data.isnull().sum()
data_summary = data.describe()
columns_list = data.columns.tolist()

# Check unique values for categorical columns
print(data['Fuel_Type'].value_counts())
print(data['Seller_Type'].value_counts())
print(data['Transmission'].value_counts())

fuel = data['Fuel_Type']
seller = data['Seller_Type']
transmission = data['Transmission']
price = data['Selling_Price']

# Plot categorical data
plt.style.use('ggplot')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Categorical Data Distribution')
axes[0].bar(fuel, price, color='royalblue')
axes[0].set_xlabel("Fuel Type")
axes[0].set_ylabel("Price")
axes[1].bar(seller, price, color='red')
axes[1].set_xlabel("Seller Type")
axes[2].bar(transmission, price, color='purple')
axes[2].set_xlabel("Transmission Type")
plt.show()

# Visualizing with seaborn
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Categorical Features vs Price')
sns.barplot(x=fuel, y=price, ax=axes[0])
sns.barplot(x=seller, y=price, ax=axes[1])
sns.barplot(x=transmission, y=price, ax=axes[2])

# Grouping data
petrol_cars = data[data['Fuel_Type'] == 'Petrol']
dealer_sellers = data[data['Seller_Type'] == 'Dealer']

# Encoding categorical variables
encoding_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
data['Fuel_Type'] = data['Fuel_Type'].map(encoding_map)
data = pd.get_dummies(data, columns=['Seller_Type', 'Transmission'], drop_first=True)

# Correlation heatmap
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()

# Scatter plot for regression
plt.figure(figsize=(7,5))
plt.title('Price vs Present Price')
sns.regplot(x='Present_Price', y='Selling_Price', data=data)

# Splitting data into training and test sets
X = data.drop(['Car_Name', 'Selling_Price'], axis=1)
y = data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Model evaluation
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
r2 = metrics.r2_score(y_test, predictions)

print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)
print("R2 Score: ", r2)

# Actual vs Predicted values
sns.regplot(x=predictions, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.title("Actual vs Predicted Selling Price")
plt.show()
