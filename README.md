# CodeClauseInternship_Movie-Genre-Prediction

# Movie Genre Prediction

This repository contains a machine learning project that predicts movie genres based on plot descriptions.

## Overview

The project utilizes natural language processing (NLP) techniques to preprocess and analyze movie plot descriptions. It then builds a machine learning model using the Multinomial Naive Bayes algorithm to predict the genre(s) of movies based on their plots.

## Dataset

The dataset (`movies.csv`) used in this project contains movie details including plot descriptions and genres. It was preprocessed to clean and tokenize the plot descriptions before training the model.
Data was sourced from [Wikipedia Movie Plots Dataset on Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)


## Dependencies

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - spacy
  - matplotlib
  - seaborn
  - joblib
  - wordcloud

## Project Structure

- `data/`: Contains the dataset (`movies.csv`) used in the project.
- `models/`: Stores the trained machine learning model (`movie_genre_classifier.pkl`).
- `notebooks/`: Jupyter notebooks used for data exploration, preprocessing, model training, and evaluation.
- `README.md`: This file, providing an overview of the project, instructions, and dependencies.

## Usage
1. **Clone the repository:**
    ```bash
    git clone https://github.com/vedant713/CodeClauseInternship_Movie-Genre-Prediction.git
    
    
  2.**Install dependencies**:     
  
      
          pip install -r requirements.txt
  
      
  3.**Navigate to notebooks/ directory**:
  
  Explore the Jupyter notebooks to understand the data preprocessing, model training, and evaluation steps.
  Run the notebooks:
  Execute the notebooks sequentially to reproduce the steps and results of the movie genre prediction project.
  Train and save the model:
  Once satisfied with the model performance, train the final model using the provided notebooks and save it using joblib.
  Predict movie genres:
  Use the trained model '(movie_genre_classifier.pkl)' to predict genres for new plot descriptions.
 
