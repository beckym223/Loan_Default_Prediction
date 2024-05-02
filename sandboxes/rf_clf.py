import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns



## TRY UNDERSAMPLING 

# Load the Dataset
df = pd.read_csv('loan_data.csv')

# Preprocess the categorical features
categorical_features = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                        'HasDependents', 'LoanPurpose', 'HasCoSigner']


# ** TO DO **
# One-hot encode categorical features ** Try without one hot encoding ** 
encoders = {}
encoded_dfs = []
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    ohe = OneHotEncoder(sparse_output=False)
    encoded_arr = ohe.fit_transform(df[feature].values.reshape(-1, 1))
    encoded_df = pd.DataFrame(encoded_arr, columns=ohe.get_feature_names_out([feature]))
    encoded_dfs.append(encoded_df)
    encoders[feature] = {'label_encoder': le, 'one_hot_encoder': ohe}

encoded_df = pd.concat(encoded_dfs, axis=1)
categorical_encoded_cols = encoded_df.columns

# Combine the one-hot encoded categorical features with the numerical features
quant_features = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                  'NumCreditLines', 'LoanTerm', 'InterestRate', 'DTIRatio']
X = pd.concat([encoded_df, df[quant_features]], axis=1)
y = df['Default']

# Apply SMOTE for oversampling the minority class
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=99)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [80, 100, 120],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create the Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best estimator and its hyperparameters
best_clf = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the best model on the test set
y_pred = best_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


