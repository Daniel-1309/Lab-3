import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)  
    
    # Drop missing values
    data.dropna(inplace=True)
    
    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)

    # Drop irrelevant features 
    if 'placement_status' in data.columns:
        data.drop('placement_status', axis=1, inplace=True)
    # Split the data into features and target variable
    X = data.drop('salary_package_lpa', axis=1)
    y = data['salary_package_lpa']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save the preprocessed data
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    return 0

if __name__ == "__main__":
    preprocess_data("data/student_salary.csv")