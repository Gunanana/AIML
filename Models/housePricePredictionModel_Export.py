# %%
# STEP 1 : Data Preparation
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("D:\IMPORTANT\Programming\AIML\Data\melbourne_data.csv")
# print(data.iloc[:5, :3])
# print(data.columns)
# print(len(data))

y = data.Price
# print(y.iloc[:10])
iFeatures = [
        'Suburb', 'Address', 'Rooms',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom',
       'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount']

X = data[iFeatures]
# print(X.iloc[0:4, :])
nan_count_per_column = X.isna().sum()
# print(nan_count_per_column)

# break into 3 sets. Train, Validation and Test data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, train_size=0.6)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state = 1, train_size = 0.5)

# %%
# STEP 2 : Transformers -> Imputer (for missing values) & OneHotEncoding (for cat data)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# OneHotEncoding
categorical_features = ['Suburb', 'Address', 'Regionname', 'Date']

# Declare a preprocessor transformer object to transform your categorical data columns
preprocessor = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_features)
    ],
    remainder = 'passthrough'
)

# fit the transformer
preprocessor.fit(X_train)

# transform your data sets
X_train = preprocessor.transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# %%
# Imputation
# Declare imputer transformer to impute the data
imputer = SimpleImputer(strategy = 'mean')

# fit the transformer
imputer.fit(X_train)

# transform your data sets
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
X_val = imputer.transform(X_val)

# %%
# STEP 3 : Validation. (get best max_leaf_nodes count)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(leaves : int):
    print(f"Training RF Model with {leaves} max_leaf_nodes")

    rfModel = RandomForestRegressor(random_state = 1, max_leaf_nodes = leaves, n_jobs=-1)
    rfModel.fit(X_train, y_train)
    return mean_absolute_error(y_val, rfModel.predict(X_val))

mae_minima = float('inf')
best_leaf_count = 0

for leaves in range(100, 1000, 250):
    mae = get_mae(leaves)
    if (mae < mae_minima):
        mae_minima = mae
        best_leaf_count = leaves

print(f"Best leaf count : {best_leaf_count}") # its 850

# %%
# STEP 4 : Testing

print("Testing the Model...")

# Declare the model
final_model = RandomForestRegressor(random_state = 1, n_jobs = -1, max_leaf_nodes = best_leaf_count)

# Fit the model
final_model.fit(X_train, y_train)

# Predict using the model
predictions = final_model.predict(X_test)

# %%
# STEP 5 : Results
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error


mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("Here are the results:")
print(f"MAE (Mean Absolute Error) : $ {mae}")
print(f"MSE (Mean Squared Error) : {mse}")
print(f"MAPE (Mean Absolute Percentage Error) : {(mape*100) :.2f}%")

# %% [markdown]
# 1. MAE of $173,588.85:
# This means your model's average prediction error is approximately $173,588.85. Depending on the range of house prices in your dataset, this could be a significant or minor error. For high-value houses, this might be more acceptable than for lower-value houses.
# 
# 2. MSE of $80,870,750,974.31:
# This large value indicates that there are some significant prediction errors. The squaring of errors in MSE means that larger errors are disproportionately impacting this metric. High MSE suggests that your model may be struggling with outliers or large variations in predictions.
# 
# 3. MAPE of 16.26%:
# A MAPE of 16.26% means that, on average, your predictions are off by about 16.26% from the actual values. For house price predictions, this could be considered moderately high. Typically, a MAPE below 10% is desirable, especially in real estate where accurate pricing is critical.


