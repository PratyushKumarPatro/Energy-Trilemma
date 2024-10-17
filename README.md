# Energy-Trilemma
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.impute import KNNImputer

file_path = 'output_file.csv'
df=pd.read_csv("output_file.csv")

continuous_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=[object]).columns

class CategoricalImputer(TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy

    def fit(self, X, y=None):
        self.imputer = SimpleImputer(strategy=self.strategy)
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return self.imputer.transform(X)

continuous_imputer = Pipeline([('imputer', SimpleImputer(strategy='mean'))])
categorical_imputer = Pipeline([
    ('imputer', CategoricalImputer())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('continuous', continuous_imputer, continuous_cols),
        ('categorical', categorical_imputer, categorical_cols)
    ])

imputed_data = preprocessor.fit_transform(df)
imputed_df = pd.DataFrame(imputed_data, columns=continuous_cols.append(categorical_cols))
imputed_df.loc[:,'m2_q69_elec_hrs']
imputed_df.dtypes

target_categorical = []
target_continuous = imputed_df.loc[:, 'm2_q69_elec_hrs']

# Use len() to get the length of the series
for i in range(len(target_continuous)):
    current_value = target_continuous.iloc[i]  # Retrieve the actual value at the current index
    if current_value == 0:
        target_categorical.append("Not_available")
    elif 1 < current_value <= 4:
        target_categorical.append("Limited_availability")
    elif 4 < current_value <= 8:
        target_categorical.append("Partially_available")
    elif 8 < current_value <= 23:
        target_categorical.append("Mostly_available")
    else:
        target_categorical.append("Fully_available")

print(target_categorical)

target = target_categorical
target

features_dropped = ["m2_q69_elec_hrs"]
features= imputed_df.drop(columns=features_dropped)
X= features
y=target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

continuous_features = ['m2_q64_alight_read_children', 'm2_q64_alight_read_adults', 'm2_q70_elec_night_hrs', 'm2_q71_elec_out_days']
categorical_features = [col for col in X.columns if col not in continuous_features]

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', [StandardScaler(), KNNImputer(n_neighbors=2)], continuous_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])
categorical_cols = features.select_dtypes(include=['object']).columns.tolist()

# One-hot encode categorical columns
encoder = OneHotEncoder(handle_unknown='ignore')
features_encoded = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

X = features_encoded
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest classification
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy without feature selection: {accuracy:.2f}")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
recall = recall_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
f1 = f1_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
conf_matrix = confusion_matrix(y_test, y_pred)

# Performance evaluation metrics
print(f"Accuracy using Random forest: {accuracy:.4f}")
print(f"Precision using Random forest: {precision:.4f}")
print(f"Recall using Random forest: {recall:.4f}")
print(f"F1 Score using Random forest: {f1:.4f}")
print("Confusion Matrix using Random forest:")
print(conf_matrix)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Define class labels
class_names = ['Not available', 'Limited availability', 'Partially available','Mostly available', 'Fully available']

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(10, 8), dpi=200)
sns.set(font_scale=1.5)  # Set font scale for labels and annotations
# Plot confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted',fontsize=15)
plt.ylabel('Actual',fontsize=15)
plt.show()


# Feature importance analysis
importances = rf_classifier.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-10:]  # Get the indices of the top 20 features

# Plot for feature importance analysis chart
plt.figure(figsize=(10, 8))
#plt.title("Top 10 Feature Importances", fontsize=15)
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), cleaned_feature_names, fontsize=30)  # Set fontsize here
plt.xlabel("Mean Importance Score", fontsize=30)
plt.show()

 # ANN Classification

pip install tensorflow

from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectFromModel

# Create the Multi-Layer Perceptron (MLP) Classifier
ann_classifier = MLPClassifier(random_state=42)

# Train the ANN classifier
ann_classifier.fit(X_train, y_train)
y_pred = ann_classifier.predict(X_test)

# Performance Metrices
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
recall = recall_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
f1 = f1_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy using Artificial neural networks: {accuracy:.4f}")
print(f"Precision using Artificial neural networks: {precision:.4f}")
print(f"Recall using Artificial neural networks: {recall:.4f}")
print(f"F1 Score using Artificial neural networks: {f1:.4f}")
print("Confusion Matrix using Artificial neural networks:")
print(conf_matrix)

plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted',fontsize=15)
plt.ylabel('Actual',fontsize=15)
plt.show()

feature_labels = [f'Feature {i+1}' for i in range(X.shape[1])]

# Plot top 10 feature importances
plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_labels)[sorted_idx], perm_importance.importances_mean[sorted_idx], color='skyblue')
plt.xlabel('Mean Importance Score')
plt.title('Top 10 Feature Importances with Custom Labels')
plt.gca().invert_yaxis()  # Invert y-axis to show the highest score on top
plt.show()


# Gradient Boosting Classification

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ColumnTransformer and Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ~features.columns.isin(categorical_cols)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
])

X = pipeline.fit_transform(features)


# Initialize a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train Classifier
gb_classifier.fit(X_train, y_train)

# Perofmance evaluation
y_pred = gb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
recall = recall_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
f1 = f1_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy using Gradient Boosting: {accuracy:.4f}")
print(f"Precision using Gradient Boosting: {precision:.4f}")
print(f"Recall using Gradient Boosting: {recall:.4f}")
print(f"F1 Score using Gradient Boosting: {f1:.4f}")
print("Confusion Matrix using Gradient Boosting:")
print(conf_matrix)

# Feature importance analysis for Gradient Boosting
importances = gb_classifier.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[-10:]  # Get the indices of the top 10 features
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align="center",color='green')
plt.yticks(range(len(indices)), cleaned_feature_names, fontsize=30)  # Set fontsize here
plt.xlabel("Mean Importance Score", fontsize=30)
plt.show()

# Softmax Regression

from sklearn.linear_model import LogisticRegression

# Softmax regression classifier
softmax_classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500, random_state=42)

# Fit the classifier on the training data
softmax_classifier.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = softmax_classifier.predict(X_test)

# Performance evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
recall = recall_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
f1 = f1_score(y_test, y_pred, average='weighted')  # Choose 'micro', 'macro', or 'weighted'
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy using Softmax regression: {accuracy:.4f}")
print(f"Precision using Softmax regression: {precision:.4f}")
print(f"Recall using Softmax regression: {recall:.4f}")
print(f"F1 Score using Softmax regression: {f1:.4f}")
print("Confusion Matrix using Softmax regression:")
print(conf_matrix)


# Feature importance analysis using the Softmax regression coefficients
importances = np.abs(softmax_classifier.coef_).mean(axis=0)  # Taking the mean of the absolute values across classes
feature_names = X.columns
indices = np.argsort(importances)[-10:]  # Get the indices of the top 10 features

# Create a plot for feature importance
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align="center",color='orange')
plt.yticks(range(len(indices)), cleaned_feature_names, fontsize=30)  # Set fontsize here
plt.xlabel("Mean Importance Score", fontsize=30)
plt.show()






