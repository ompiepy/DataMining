# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# %%
df  = pd.read_csv('voice.csv')

# %%
df.head(5)

# %%
df.columns

# %%
df['label'].unique()

# %%
df['label']

# %%
df['label_encoded'] = df['label'].apply(lambda x: 1 if x == 'female' else 0)
df['label_encoded']

# %%
df.head(5)

# %%
df.min()

# %%
df.drop('label_encoded',axis=1).describe()

# %%
df.hist(figsize=(20,20))
plt.show()

# %%
from sklearn.preprocessing import StandardScaler

# %%
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df.drop(['label','label_encoded'], axis=1))

# %%
columns = df.columns.tolist()
columns.remove('label')
columns.remove('label_encoded')

# %%
scaled_df = pd.DataFrame(scaled_df, columns=columns)

# %%
scaled_df.head()

# %%
scaled_df.hist(figsize=(20, 20))
plt.show()

# %%
scaled_df.describe()

# %%
X = scaled_df
y = df['label_encoded']

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# %%
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

# %%
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %%
from sklearn.metrics import classification_report, confusion_matrix

# %%
print(classification_report(y_test, y_pred))

# %%
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Blues', annot=True, fmt='.0f')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# %%
# findinfg best value of spectrum
from sklearn.model_selection import cross_val_score
import numpy as np

k_scores = []
k_range = range(1, 50)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    
best_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)
# Plot the cross-validation scores
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Finding the best k')
plt.grid(True)

plt.axvline(best_k, color='r', linestyle='--')
plt.annotate(f'Best k = {best_k}\nAccuracy = {best_score:.4f}',
             xy=(best_k, best_score), xytext=(best_k, best_score + 0.02),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# Find the best k
best_k = k_range[np.argmax(k_scores)]
print(f'The best value for k is {best_k} with a cross-validated accuracy of {max(k_scores):.4f}')

# %%
corr = df.drop('label', axis=1).corr()
corr.iloc[:-1,-1]

# %%
final_df = scaled_df.copy()
final_df['label'] = y

# %%
final_df.head()

# %%
sns.pairplot(final_df)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_dataset.csv' with the path to your dataset)

# Calculate the correlation matrix
correlation_matrix = pd.concat([X, y.rename('Class')], axis=1).corr()

# Extract the correlation of the target class with other features (replace 'class' with your target column name)
target_correlation = correlation_matrix['Class']

# Drop the target class correlation with itself
target_correlation = target_correlation.drop('Class')

# Plot the correlation values
plt.figure(figsize=(10, 8))
sns.heatmap(target_correlation.to_frame(), annot=True, cmap='coolwarm', cbar=True, vmin=-1, vmax=1)
plt.title('Correlation of Class with All Other Features')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
plt.show()


# %%
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')

# %%

data = pd.concat([X, y.rename('Class')], axis=1)

# Compute the correlation matrix
corr = data.corr()

# Extract the correlations between the features and the class variable
class_corr = corr['Class'].drop('Class')

# Plot the correlations using a heatmap
plt.figure(figsize=(20, 16))
sns.barplot(x=class_corr.index, y=class_corr.values, color='blue')
plt.title('Pearson Correlation between Features and Class Variable')
plt.xlabel('Class Variable')
plt.ylabel('Features')
plt.yticks(rotation=0)  # Rotate feature names for better readability
plt.show()

# %%
sorted_corr = data.corr()['Class']
sorted_corr = sorted_corr.sort_values(ascending=False)[1:6]
plt.figure(figsize=(10, 8))
sns.barplot(x=sorted_corr.index, y=sorted_corr.values, color='blue')
plt.title('Pearson Correlation between Features and Class Variable')
plt.xlabel('Class Variable')
plt.ylabel('Features')
plt.yticks(rotation=0)  # Rotate feature names for better readability
plt.show()


# %%
sorted_corr = data.corr()['Class']
sorted_corr = sorted_corr.sort_values(ascending=True)[1:6]
plt.figure(figsize=(10,8))
sns.barplot(x=sorted_corr.index, y=sorted_corr.values, color='blue')
plt.title('Pearson Correlation between Features and Class Variable')
plt.xlabel('Class Variable')
plt.ylabel('Features')
plt.yticks(rotation=0)  # Rotate feature names for better readability
plt.show()

# %%
from sklearn.neighbors import KNeighborsClassifier
weighted_knn = KNeighborsClassifier()

# %%
weighted_knn

# %%
weighted_knn.fit(X_train, y_train)

# %%
random_weights = np.random.rand(X_train.shape[1])

# Multiply the features by these weights
X_train_weighted = X_train * random_weights

# Define the range of k values
k_scores = []
k_range = range(1, 50)

# Perform cross-validation for each k value
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_weighted, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# Generate random weights for each feature
np.random.seed(10)
random_weights = np.random.rand(X_train.shape[1])

knn = KNeighborsClassifier( 
    metric_params={'w': random_weights}
)

# Perform cross-validation
scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

# Fit the model and print the cross-validated accuracy
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Cross-validated accuracy:", scores.mean())

# Predict and print the test accuracy
test_accuracy = knn.score(X_test, y_test)
print("Test accuracy:", test_accuracy)
knn

# %%
print(classification_report(y_pred,y_test))

# %%
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='.0f')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score


# Generate random weights for each feature
np.random.seed(10)
random_weights = np.random.rand(X_train.shape[1])

knn = KNeighborsClassifier( 
    metric_params={'w': random_weights}
)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
knn

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# %%
print(classification_report(y_test, y_pred))

# %%
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='.0f')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


