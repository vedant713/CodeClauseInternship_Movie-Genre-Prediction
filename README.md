# CodeClauseInternship_Movie-Genre-Prediction

# Movie Genre Prediction

This repository contains a machine learning project that predicts movie genres based on plot descriptions.

## Overview

The project utilizes natural language processing (NLP) techniques to preprocess and analyze movie plot descriptions. It then builds a machine learning model using the Multinomial Naive Bayes algorithm to predict the genre(s) of movies based on their plots.

## Dataset

The dataset (`movies.csv`) used in this project contains movie details including plot descriptions and genres.Data was sourced from [Wikipedia Movie Plots Dataset on Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots).It was preprocessed to clean and tokenize the plot descriptions before training the model.

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

