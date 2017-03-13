## Daily sales predictions

The aim of this project is to predict daily sales in advance for Rossmann drug stores located in Germany. The project is based on a kaggle competition. The description of the competition can be found [here](https://www.kaggle.com/c/rossmann-store-sales). Store sales are influenced by many factors such as promotions, competition, school and state holidays, seasonality, and locality etc. The idea is to use these features and build a machine learning mode to predict sales in advance so that store managers can prepare and focus on what's most important to them.

#### Installation
- Clone this repo to your computer
- Change directory `cd predict_sales`
- Run `mkdir data` and download the csv data files from [here](https://www.kaggle.com/c/rossmann-store-sales/data) into this directory
- Install the requirements using `pip install -r requirements.txt` if moduels are not already exisiting 

#### Files
- **exploratory_analysis.ipynb** : A python nobebook for exploring the data and building features as necessary
- **predict_sales.py** : Script to make predictions. It prints the values cross-validation score (accuracy) and RMSPE. 
- **sales.csv** : Output of the predict_sales.py script containing actual sales and corresponding store IDs
- **README.md** : You are reading this file
- **data** : This is the data folder, not included here. The data can be downloaded from this [link](https://www.kaggle.com/c/rossmann-store-sales/data). A description of the data fields are also present at this link.

#### Summary
We built a machine learning model to predict daily sales of 1115 Rossmann drug stores. The accuracy of prediction is about 85%. This accuracy was achieved using only four features - *'Store', 'DayOfWeek', 'Promo', 'Year'*. Additional features do not improve the model very much, infact they worsen the accuracy of prediction. It looks like store promotions have biggest impact on daily sales. The data also  

In this model have not made use of the time-dependent nature of the data very much, except the *'Year'* feature. Since store sales can depend on short-term history, one can use tools of time-series analysis to further study and possibly improve the model. 

This model can be easily embbeded in a web application (for example, using python Flask and AWS) which can be used by store managers to plan ahead with their inventory requirements.   
 
