# ForexML
Classical and Deep-Learning Methods for Time-Series Forecasting of Forex Data


This Readme file is designed for preprocessing financial time series data and applying three different algorithms, specifically for Forex price prediction tasks. The preprocessing includes loading a CSV file, dropping unnecessary columns, normalizing the "Volume" column, scaling the "High," "Low," and "Close" columns, and creating feature-target pairs for training a machine learning model.

# Dependencies

Make sure you have the following libraries installed:
numpy 
pandas
dask.dataframe
statsmodels.api 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Usage
## Preprocessing

1. Load CSV File
Modify the `directory` variable to point to your CSV file.
Use the `pd.read_csv` function to load the data into a pandas DataFrame.

2. Remove unnecessary columns (e.g., "Date," "Time," "Open") from the DataFrame.

3. Normalize and Scale:
The script contains a function scale_sequence that normalizes the "Volume" column and scales "High," "Low," and "Close" columns for each sequence.

4. Create Feature-Target Pairs:
The script generates feature-target pairs for training a machine learning model.
Adjust the window_size and future_steps parameters according to your requirements.
5. Convert to NumPy Arrays:
Convert the lists of features, targets, and MaxMin to NumPy arrays.

6. Define Test Size and Random State:
Adjust the “test_size” parameter to set the proportion of the dataset used for testing.
Set the “random_state” parameter to a specific value for reproducibility.
 python test_size = 0.2 
Adjust as needed random_state = 42 

Set a random state for reproducibility

7. Split Features and Targets:
Use the train_test_split function to split the features and targets into training and testing sets.
8. Shuffle Training Set:
Shuffle the training set to enhance the model's learning process.
9. Shuffle MaxMin Values:
Shuffle the corresponding MaxMin values to maintain consistency.


## ARIMA
- Ensure that the preprocessing code was run before running the ARIMA code. 
- Training and Prediction 
- Runs for 1000 iterations 
- Randomly chooses an index to select a random entry.
- Updates two models with new data and makes predictions on the dataset based on the models.
- Calculates MSE for predictions 
- Collect all predictions in ‘ypredict’
- Testing 
- Each row of ‘Xtest’ it does the following
- Generates predictions for two features based on the total number of best models and computes the mean
- Updates the temporary array with predictions
- Computes the mean of predictions and appends to ‘ypredicttest’


## Ridge Regression
- Ensure that the preprocessing code was run before running the Ridge Regression code. 
- Generate a range of alpha values, and initialize empty lists.
- For each alpha value
- Creates a ridge regression model 
- Train the model on the training data
- Predict the labels 
- Calculate the MSE for the training model and append 
- Evaluate the model on the testing data
- Calculate MSE for testing model and append 

## DNN
- Ensure that the preprocessing code was run before running the Ridge Regression code. 
- Split the data into a train and test set.
- To run the models you may need a high RAM, since they’re using large memories.
- It’s suggested to run the models on a GPU since training them is very time consuming (it will take around half an hour). Note that not all GPUs are suitable for all the models.
- There are different models that can be used
	- CNN-LSTM
	- LSTM-CNN
	- LSTM and CNN in parallel
	


