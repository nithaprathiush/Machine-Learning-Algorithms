from shiny import App, ui, render
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
    

app_ui = ui.page_fluid(

    ui.h1("Decision Tree Classifiication Problem"),
    ui.h1("_____________________________________"),
    ui.h4("DataSet : The breast cancer dataset in scikit-learn helps doctors diagnose tumors in the breast. It includes information from images of cells, like their size and shape. With 569 records and 31 columns for each, it's perfect for deciding if a tumor is cancerous or not. Using decision tree algorithm , created a model to predict the diagnosis."),
    ui.h3("Dataset Size"),
    ui.h4(ui.output_text("LoadData")),
    ui.h3("Accuracy: "),
    ui.h4(ui.output_text_verbatim("ModelTraining")),
    ui.h4("To improve the accuracy, Incorporated parameter tuning using GridSearchCV."),
    ui.h3("Accuracy: "),
    ui.h4(ui.output_text_verbatim("ParamerTuning")),

    ui.row(
        ui.output_plot("plotGraph")      
    ),
)

def server(input, output, session):

    @render.text  
    def LoadData():
        # Load breast cancer dataset
        data = load_breast_cancer()
        # Convert to DataFrame
        df_cancer = pd.DataFrame(data.data, columns=data.feature_names)
        df_cancer['target'] = data.target
 
        return df_cancer.shape
    
    @render.text 
    def ModelTraining():

          # Load breast cancer dataset
        data = load_breast_cancer()
        # Convert to DataFrame
        df_cancer = pd.DataFrame(data.data, columns=data.feature_names)
        df_cancer['target'] = data.target
        X = df_cancer.drop(columns=['target'])
        Y = df_cancer['target']
        # Split the dataset into training and testing sets:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        # Initialize a decision tree classifier
        clf = DecisionTreeClassifier()
        #Fit the classifier to the training data:
        clf.fit(X_train, y_train)
        # Use the trained model to make predictions on the test set
        y_pred = clf.predict(X_test)
        # Assess the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        # print("Accuracy:", accuracy) 

        return accuracy
    
    @render.text  
    def ParamerTuning():

          # Load breast cancer dataset
        data = load_breast_cancer()
        # Convert to DataFrame
        df_cancer = pd.DataFrame(data.data, columns=data.feature_names)
        df_cancer['target'] = data.target
        X = df_cancer.drop(columns=['target'])
        Y = df_cancer['target']
        # Split the dataset into training and testing sets:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

                # Define the parameter grid
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

         # Initialize a decision tree classifier
        clf = DecisionTreeClassifier()
        #Fit the classifier to the training data:
        clf.fit(X_train, y_train)

        # Perform grid search
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_

        # Train the model with the best parameters
        best_clf = DecisionTreeClassifier(**best_params)
        best_clf.fit(X_train, y_train)
        # Make predictions
        y_pred = best_clf.predict(X_test)

        # Evaluate the model
        accuracy_Tuned = accuracy_score(y_test, y_pred)

        return accuracy_Tuned
    
    @render.plot  
    def plotGraph():

          # Load breast cancer dataset
        data = load_breast_cancer()
        # Convert to DataFrame
        df_cancer = pd.DataFrame(data.data, columns=data.feature_names)
        df_cancer['target'] = data.target
        X = df_cancer.drop(columns=['target'])
        Y = df_cancer['target']
        # Split the dataset into training and testing sets:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

                # Define the parameter grid
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

         # Initialize a decision tree classifier
        clf = DecisionTreeClassifier()
        #Fit the classifier to the training data:
        clf.fit(X_train, y_train)

        # Perform grid search
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best parameters
        best_params = grid_search.best_params_

        # Train the model with the best parameters
        best_clf = DecisionTreeClassifier(**best_params)
        best_clf.fit(X_train, y_train)
        # Make predictions
        y_pred = best_clf.predict(X_test)

        # Evaluate the model
        accuracy_Tuned = accuracy_score(y_test, y_pred)
        
        # Plot the decision tree

        fig, ax = plt.subplots(figsize=(20, 30))
        plot_tree(best_clf, filled=True, feature_names=data.feature_names, class_names=data.target_names, ax=ax)
        return fig    
             
app = App(app_ui, server)
