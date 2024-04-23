import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import joblib


df = pd.read_csv("./Datasets/train.csv") ## Load Dataset

## Categorising into numerical & categorical features
numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

## Mode Imputation of Categorical cols
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

## Median Imputation of Numerical cols
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

## Outliers Handling
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# Log Transforamtion & Domain Processing
df['LoanAmount'] = np.log(df['LoanAmount']).copy()
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'] = np.log(df['TotalIncome']).copy()


## Dropping ApplicantIncome and CoapplicantIncome
df = df.drop(columns=['ApplicantIncome','CoapplicantIncome'])

## Label encoding categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

#Encode  target column
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# Train test split
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df.Loan_Status
RANDOM_SEED = 6

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, random_state = RANDOM_SEED)

# RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200,400, 700],
    'max_depth': [10,20,30],
    'criterion' : ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_forest, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=0
    )
model_forest = grid_forest.fit(X, y)

joblib.dump(model_forest, './Trained-Models/RF_Loan_model.pkl')

print("Welcome to Loan Prediction Application")
print("Enter your details to get to know the application status as per the instruction below:")
print("Type 'exit' to terminate.....\n")
print('''Gender: Female = 0, Male=1
Married: No = 0, Yes = 1
Education: Graduate = 0 , Under-graduate = 1
Self_Employed: No = 0, Yes = 1
Property_Area: Urban = 2, Semiurban = 1, Rural = 0
Loan_Status: No = 0, Yes = 1\n''')

print('''Enter the data with the below mentioned order , where values are seperated by comma: 
Gender, Married, Dependents,Education,Self_Employed,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area,TotalIncome\n''')

while True:
    user_data = input("Enter your Details: ")
    
    if(user_data == "exit"):
        break

    data = list(map(float, user_data.split(','))) ## Convert to float

    # basic validation
    if(len(data)<10):
        print("Incomplete data provided!!")
    else:
        
        # predicting the value
        predicted_value=model_forest.predict([data])
        print("/**********************************************************************/")
        if (predicted_value[0]):
            print("\tCongratulations! your loan request is Approved and the output is: ",predicted_value[0])
        else:
            print("\tSorry! your loan request is Rejected, please try again after 6 months and the output is: ",predicted_value[0])
        print("/**********************************************************************/")