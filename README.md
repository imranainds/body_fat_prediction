# Body Fat Prediction

This repository contains a Python implementation of a machine learning model for predicting body fat percentage from various anthropometric measurements. The model utilizes a Random Forest regressor and is trained on a dataset of over 10,000 individuals.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

- Python 3.7 or later
- Pandas 1.0.3 or later
- Numpy 1.18.2 or later
- Scikit-learn 0.22.1 or later
- Matplotlib 3.2.1 or later

### Installation

1. Clone the repository:


bash
git clone https://github.com/yourusername/bodyfatprediction.git
​


2. Change to the project directory:


bash
cd bodyfatprediction
​


3. Install the required Python packages:


bash
pip install -r requirements.txt
​


## Usage

To use the model for predicting body fat percentage, follow these steps:

1. Import the necessary libraries:


python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
​


2. Load the dataset:


url = https://github.com/imranainds/body_fat_prediction/blob/main/bodyfat.csv
df = pd.read_csv(url)
​


3. Preprocess the data:



# Split the dataset into features (X) and target (y)
X = df.drop('bodyfat', axis=1)
y = df['bodyfat']
​
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
​


4. Train the model:


# Create a Random Forest regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
​
# Train the model on the training data
model.fit(X_train, y_train)
​


5. Make predictions:


# Make predictions on the test data
y_pred = model.predict(X_test)
​


6. Evaluate the model:


# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
​
# Plot the actual vs predicted body fat percentage
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Body Fat Percentage')
plt.ylabel('Predicted Body Fat Percentage')
plt.title('Actual vs Predicted Body Fat Percentage')
plt.show()
​


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project is sourced from the [Body Composition Data Sets](https://archive.ics.uci.edu/ml/datasets/Body+Composition) page at the University of California, Irvine.
- The model's performance can be further improved by experimenting with different machine learning algorithms, feature engineering techniques, and hyperparameter tuning.
- Please note that this project is for educational purposes only and should not be used for medical or health-related advice. Consult a healthcare professional for personalized advice.