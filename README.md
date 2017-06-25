## Daily sales predictions

The aim of this project is to predict daily sales in advance for Rossmann drug stores located in Germany. The project is based on a kaggle competition. The description of the competition can be found [here](https://www.kaggle.com/c/rossmann-store-sales). Store sales are influenced by many factors such as promotions, competition, school and state holidays, seasonality, and locality etc. The idea is to use these features and build a machine learning model to predict sales in advance so that store managers can prepare and focus on what's most important to them.

#### Installation
- Clone this repo to your computer
- Change directory `cd predict_sales`
- Run `mkdir data` and download the csv data files from [here](https://www.kaggle.com/c/rossmann-store-sales/data) into this directory
- Install the requirements using `pip install -r requirements.txt` if moduels are not already existing 

#### Files
- **exploratory_analysis.ipynb** : A python nobebook for exploring the data and building features as necessary
- **predict_sales.py** : Script to make predictions. It prints the values cross-validation score (accuracy) and RMSPE. 
- **sales.csv** : Output of the predict_sales.py script containing actual sales and corresponding store IDs
- **README.md** : You are reading this file
- **data** : This is the data folder, not included here. The data can be downloaded from this [link](https://www.kaggle.com/c/rossmann-store-sales/data). A description of the data fields are also present at this link.

#### Summary
We built a model to predict daily sales of 1115 Rossmann drug stores across Germany. We tried different regression models such as Linear Regressor, GradientBoost and AdaBoost and found that Random Forest Regressor does better than others. The evaluation metric for this task is root mean squared percentage error ([RMSPE](https://www.kaggle.com/c/rossmann-store-sales#evaluation)). The best RMSPE of 12.6% was achieved with Kaggle's test data set. This error was obtained using only four features - **'Store', 'DayOfWeek', 'Promo', 'Year'**. Additional features do not improve the model very much, in fact they worsen the accuracy of prediction. The analysis show that store promotions have biggest impact on daily sales. 

In this model have not made use of the time-dependent nature of the data, except the *'Year'* feature. A quick check (using lag and auto-correlation plots) for two stores did not reveal any strong auto-correlations in the sales data, but store sales can sometimes depend on short-term history, so one can further explore the tools of time-series analysis to this problem. 

This model can be easily embedded in a web application (for example, using python Flask with AWS) which can be used by store managers to plan ahead with their inventory requirements, staff assignments and other arrangements that depend on the magnitude of sales at any given day, and thereby improve the overall productivity.
 