# Stack Overflow Tag Prediction

## Project Overview

This project aims to build a machine learning model to predict the most relevant tag for a new post on Stack Overflow. The tags predicted by this model are limited to the following nine: Python, JavaScript, iOS, C#, Java, Ruby-on-Rails, MySQL, HTML, and Matplotlib. 

The model was trained using a limited sample of posts from stackoverflow.com, with each post having one to five tags. For this project, the tags were pared down to the nine specified tags. 

## Contents

The project contains a Jupyter notebook that has been converted into a Python script, which includes:

- Data Loading and Exploration: Loading the dataset, exploring its structure, and summarizing its properties.
- Data Preprocessing: Transforming the raw data into a form suitable for training a machine learning model.
- Data Visualization: Plotting the evolution of tags over time.
- Model Training: Fine-tuning a pre-trained DeBERTa model for batch multi-label classification and a XtremeDistill model for near-real-time predictions if needed.
- Model Evaluation: Evaluating the model's performance on a test dataset.
- Hyperparameter Tuning: Optimizing the model's hyperparameters using Optuna.
- Prediction: Using the model to predict tags for new posts.

## Usage

1. Clone the repository.
2. Install the necessary libraries specified in the requirements.txt.
3. Run the Jupyter notebook. 

## Dependencies

The Python script relies on the following libraries:

- `pandas` for data manipulation.
- `numpy` for numerical operations.
- `matplotlib` for data visualization.
- `sklearn` for computing metrics.
- `transformers` for using transformer models (specifically, the DeBERTa model).
- `datasets` for handling the dataset.
- `torch` for handling tensors.
- `wandb` for tracking experiments.
- `optuna` for hyperparameter optimization.

## Model

The model used in this project is a pre-trained DeBERTa model from Hugging Face's transformer library, fine-tuned for multi-label classification. The model was trained using a binary cross-entropy loss function.

## Evaluation

The model's performance was evaluated using the F1-score (micro-averaged), ROC AUC (micro-averaged), and accuracy. These metrics were chosen because they are well-suited for multi-label classification problems.
You can check the Weight and Biases [report here](https://api.wandb.ai/links/gaceladri/szrfi65q)
![Screenshot of the report](/media/W%26B_Report.png)

## Hyperparameter Tuning

Hyperparameters were tuned using the Optuna library. The learning rate, batch size, and weight decay were optimized to maximize the evaluation metrics.

## Limitations and Future Work

This project is a proof-of-concept and has some limitations. The model might not generalize well to other tags outside of the nine specified tags. In addition, the model does not consider the temporal information in the dataset.

In the future, more tags can be included to make the model more versatile. Other types of models and features (like the time of posting) can also be explored. Finally, deploying the model into a production environment and integrating it with the Stack Overflow platform would be the ultimate goal.

## Contact

For any questions or suggestions, please feel free to reach out.

*Note: This README is a template and should be edited to match the specifics of your project and setup.*