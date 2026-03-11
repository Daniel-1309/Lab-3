import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data_path):
    # Load the dataset
    data = pd.read_csv("../data/student_salary.csv")  
    
    # Handle missing values (example: fill with mean)
    data.fillna(data.mean(), inplace=True)
    
    # Encode categorical variables (example: one-hot encoding)
    data = pd.get_dummies(data, drop_first=True)
    
    # Split the data into features and target variable
    X = data.drop('target', axis=1)  # Replace 'target' with your actual target column name
    y = data['target']  # Replace 'target' with your actual target column name
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test