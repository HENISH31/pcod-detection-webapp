import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Step 2: Load your dataset
df = pd.read_csv('PCOD-10.csv')  # Update the file path as needed

# Step 3: Separate features (X) and target labels (y)
X = df.drop('PCOS (Y/N)', axis=1)  # Replace 'PCOS (Y/N)' with your target column name
y = df['PCOS (Y/N)']  # Target column

# Step 4: Handle missing values
imputer = SimpleImputer(strategy='mean')  # Replace 'mean' with 'median' or 'most_frequent' as needed
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Initialize the SVC model and train it
svc = SVC(kernel='linear', probability=True, random_state=42)  # Set probability=True to enable predict_proba
svc.fit(X_train, y_train)

# Step 8: Evaluate the model's performance
y_pred_test = svc.predict(X_test)

# Calculate testing accuracy
testing_accuracy = accuracy_score(y_test, y_pred_test)
print(f'Testing Accuracy: {testing_accuracy * 100:.2f}%')

# Classification report (on test set)
print("Classification Report (Test Set):")
print(classification_report(y_test, y_pred_test))

# Confusion Matrix (on test set)
print("Confusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_test))

# Step 9: Taking user input for prediction with "R" or "I" validation
def get_user_input(feature_names):
    """
    Prompts the user to input values for all features in the dataset.
    :param feature_names: List of feature names.
    :return: Numpy array of user input.
    """
    input_data = []
    print("Enter the values for the features:")

    # Define Blood Group mapping
    blood_group_mapping = {
        "A+": 11,
        "A-": 12,
        "B+": 13,
        "B-": 14,
        "O+": 15,
        "O-": 16,
        "AB+": 17,
        "AB-": 18
    }

    # Define Y/N mapping for categorical features
    yn_mapping = {
        "Y": 1,
        "N": 0
    }

    for feature in feature_names:  # Dynamically reference the feature names
        while True:
            if feature == "Blood Group":  # Check if the current feature is "Blood Group"
                value = input(f"{feature} (A+, A-, B+, B-, O+, O-, AB+, AB-): ").strip()  # Prompt for blood group
                if value in blood_group_mapping:
                    input_data.append(blood_group_mapping[value])  # Map to the corresponding numeric value
                    break
                else:
                    print("Invalid input! Please enter a valid blood group (A+, A-, B+, B-, O+, O-, AB+, AB-).")
            elif feature == "Cycle(R/I)":  # Replace with actual feature names requiring R or I
                value = input(f"{feature} (R/I): ").strip().upper()  # Prompt for 'R' or 'I'
                if value == "R":
                    input_data.append(2)  # Map 'R' to 2
                    break
                elif value == "I":
                    input_data.append(5)  # Map 'I' to 5
                    break
                else:
                    print("Invalid input! Please enter 'R' or 'I'.")
            elif feature == "Weight gain(Y/N)":  # Handling 'Weight gain (Y/N)' feature
                value = input(f"{feature} (Y/N): ").strip().upper()  # Prompt for 'Y' or 'N'
                if value in yn_mapping:
                    input_data.append(yn_mapping[value])  # Convert 'Y' to 1 and 'N' to 0
                    break
                else:
                    print("Invalid input! Please enter 'Y' or 'N'.")
            elif feature in ["Pregnant(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)",
                             "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)"]:  # Features with Y/N inputs
                value = input(f"{feature} ").strip().upper()  # Prompt for 'Y' or 'N'
                if value in yn_mapping:
                    input_data.append(yn_mapping[value])  # Convert 'Y' to 1 and 'N' to 0
                    break
                else:
                    print("Invalid input! Please enter 'Y' or 'N'.")
            else:
                try:
                    value = float(input(f"{feature}: "))  # Prompt for numerical value
                    input_data.append(value)
                    break
                except ValueError:
                    print("Invalid input! Please enter a numerical value.")

    return np.array(input_data).reshape(1, -1)

# Get user input
user_input = get_user_input(X.columns)

if user_input is not None:  # Proceed only if input is valid
    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Make prediction
    user_prediction = svc.predict(user_input_scaled)

    # Get prediction probability
    user_prediction_proba = svc.predict_proba(user_input_scaled)

    # Get the probability of having PCOS (assuming PCOS is labeled as 1)
    pc_pos_probability = user_prediction_proba[0][1]  # Probability of class 1
    pc_neg_probability = user_prediction_proba[0][0]  # Probability of class 0

    # Display the prediction and probabilities
    print(f'Prediction for the entered data: {"PCOS" if user_prediction[0] == 1 else "No PCOS"}')
    print(f'Chance of having PCOS: {pc_pos_probability * 100:.2f}%')
    print(f'Chance of not having PCOS: {pc_neg_probability * 100:.2f}%')

   # Define the folder name
folder_name = 'model'

# Create the folder if it doesn't exist
os.makedirs(folder_name, exist_ok=True)

# Save the trained SVM model
joblib.dump(svc, os.path.join(folder_name, 'pcos_svm_model.pkl'))

# Save the scaler
joblib.dump(scaler, os.path.join(folder_name, 'scaler.pkl'))

# Save the imputer
joblib.dump(imputer, os.path.join(folder_name, 'imputer.pkl'))

print(f"Model, scaler, and imputer saved successfully in the '{folder_name}' folder.")