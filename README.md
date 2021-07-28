# Stou Case Study Analysis

## OUTLINE

Case Study Data Analysis on Fraud Detection from Transaction Data. Uses Machine Learning to predict whether a transaction is fraudulent or not. Deployed to production as a Flask-based web application. 

## TECHNOLOGY STACK
- Python 3.9 - sklearn, pandas, numpy, scipy
- Flask - Web Server 
- HTML/CSS -User Interface
- Excel - for charts and visualizations

## READ ME 

- initialize a git repository in an empty folder in your system `git init`
- git clone `https://github.com/pk9444/StoutCaseStudyAnalysis`
- download the dataset form https://www.kaggle.com/ealaxi/paysim1
- save in your project directory
- open the project in pycharm/vscode or the python shell , anyway you like
- open `ml_pipeline.py` and modify the `path` variable to absolute path to your dataset
- uncomment the lines at the end of `ml_pipeline.py` - this dump classifier objects containing ML pipelines into a pickle file to be used for prediction
- run `ml_pipeline.py`, two `.pkl` files will be generated in your project directory
- now run `app.py` 
- Flask server starts up and open the link or copy paste the address into your browser
- Explore the web application, test your own custom precitions on the /predictions page and explore other findings of the case study
