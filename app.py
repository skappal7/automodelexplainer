import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, Binarizer
import pickle

# Load test data
test_data_path = '/mnt/data/Call Center Data 2022 - 2024.csv'
test_data = pd.read_csv(test_data_path)

# Prepare the data
test_data = test_data.dropna()  # Drop any rows with missing values

# Encode categorical variables
label_encoders = {}
for column in test_data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    test_data[column] = label_encoders[column].fit_transform(test_data[column])

# Binarize 'First Call Resolution (FCR %)' for classification
y_class = test_data['First Call Resolution (FCR %)']
binarizer = Binarizer(threshold=y_class.median()).fit(y_class.values.reshape(-1, 1))
y_class = binarizer.transform(y_class.values.reshape(-1, 1)).ravel()

# Split into training and testing sets for classification
X = test_data.drop(columns=['First Call Resolution (FCR %)', 'CSAT (%)'])
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Train a RandomForest model (classifier)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_class, y_train_class)

# Save the classifier model
classifier_model_path = '/mnt/data/rf_classifier_model.pkl'
with open(classifier_model_path, 'wb') as f:
    pickle.dump(rf_classifier, f)

# Split the data into features and target for regression
y_reg = test_data['CSAT (%)']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

# Train a RandomForest model (regressor)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_reg, y_train_reg)

# Save the regressor model
regressor_model_path = '/mnt/data/rf_regressor_model.pkl'
with open(regressor_model_path, 'wb') as f:
    pickle.dump(rf_regressor, f)

classifier_model_path, regressor_model_path
