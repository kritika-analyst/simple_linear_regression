# Simple Linear Regression on Housing Prices

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as mse, r2_score

df= pd.read_csv("california_housing.csv")    # load dataset
print(df.head())     # display 5 rows of the dataset

print(df.isnull().sum())    # checks and count empty values in the dataset

x= df[["AveRooms"]]   # input feature
y= df["MedInc"]       # target variable

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state= 1)  # splits the data into training and testing sets where testing data is 20%

model= LinearRegression()            # creates Linear Regression model
model.fit(x_train, y_train)          # trains the model
predictions= model.predict(x_test)   # makes predictions


mse= mse(y_test, predictions)       # calculates mean squared error
r2= r2_score(y_test, predictions)   # calculates performance of the model 

print("Mean Squared Error: ", mse)  
print("R2 Score: ", r2)

print("Intercept: ", model.intercept_)  # prints the intercept of the model
print("Coefficients: ", model.coef_)    # print the coefficients of the model

plt.figure(figsize= (8,5))  # sets the size of the plot 

plt.scatter(x_test, y_test, label= "Actual Values")        # plots actual values
plt.plot(x_test, predictions, color= "red" ,linewidth= 2)  # plots the regression line

plt.xlabel("Average Rooms")    # label for x-axis
plt.ylabel("Median Income")    # label for y-axis

plt.title("Simple Linear Regression")  # title of the plot

plt.legend()  # adds legend to the plot 

plt.savefig("linear_regression_graph.png")
plt.show()    # displays the plot

print("Model Training Completed Successfully")

print("Number of Training Samples: ",len(x_train))  # prints the number of training samples
print("Number of Testing Samples :",len(x_test))    # prints the number of testing samples 