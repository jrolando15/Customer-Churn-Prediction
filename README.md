# Customer-Churn-Prediction

#Project Description
This project aims to predict customer churn using a machine learning model. Customer churn refers to the phenomenon where customers stop doing business with a company. By predicting churn, businesses can take proactive measures to retain customers and improve their services.

#Table of Content
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Result](#result)
- [License](#license)

# Installation
To set up the project on your local machine, follow these steps:
1. Clone the respository
```bash
git clone https://github.com/your_username/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create the virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. install required dependencies
```bash
pip install -r requirements.txt
```

# Usage 
1. load the dataset
```python
df = pd.read_csv(r"path.to.the_csv.file")
```

2. Hadling Missing values
```python
missing_values = df.isnull().sum()
print(missing_values)
df.fillna(df.mean(), inplace=True)
```

3. Encoded categorical variables
```python
df = pd.get_dummies(df, columns=['Age Group', 'Tariff Plan', 'Status'])
```

4. Scale numerical features
```python
numerical_cols = ['Call Failure', "Subscription Length", 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Distinct Called Numbers', 'Age', 'Customer Value']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
```

5. Split the data into training and testing sets
```python
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

# Project Structure 
```bash
customer-churn-prediction/
├── data/
│   └── datalab_export_2024-05-25 09_10_12.csv  # Dataset file
├── churn_prediction.ipynb                       # Jupyter notebook with the code
│                    
├── README.md                                   # Project README file
└── requirements.txt                            # List of dependencies
```

# Data Preprocessing
The dataset is loaded and preprocessed using the following steps:
- Handle missing values by filling them with the mean of the respective columns.
- Encode categorical variables into dummy variables.
- Scale numerical features using StandardScaler.

# Model Training 
A Random Forest Classifier is used to train the model:
```python
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

# Model Evaluation
The model's performance is evaluated using a confusion matrix and classification report:
```python
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
cr = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(cr)
```

# Result 
The confusion matrix and classification report provide insights into the model's performance:
```python
Confusion Matrix:
[[15  0]
 [ 2  3]]

Classification Report:
              precision    recall  f1-score   support

           0       0.88      1.00      0.94        15
           1       1.00      0.60      0.75         5

    accuracy                           0.90        20
   macro avg       0.94      0.80      0.84        20
weighted avg       0.91      0.90      0.89        20
```
The confusion matrix and classification report indicate that the model has an overall accuracy of 90%. The precision, recall, and F1-score for both classes (churn and non-churn) are also provided.

# License
This README file provides a comprehensive overview of the project, including installation instructions, usage, project structure, data processing, model training, and evaluation. It also includes a license section to specify the project's licensing terms.
