# Model Card

For additional information, see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf.

## Model Details
The model in this project uses a Random Forest Classifier to predict whether an individual makes more than $50K (fifty thousand dollars) per year based on census data.

## Intended Use
The model is strictly for educational purposes to demonstrate how to build and deploy an ML pipeline using FastAPI and scikit-learn.

## Training Data
The training data comes from the UCI Census Income dataset. The data contains demographics and individual information such as employment, age, workclass, and marital status.

## Evaluation Data
The evaluation data was created using an 80/20 split of the original dataset into training and test sets.

## Metrics
The metrics used were precision, recall, and F1 score, and are as follows:
-Precision: 0.7448
-Recall: 0.6353
-F1 Score: 0.6857

## Ethical Considerations
The model uses demographic and personal information so that bias may affect predictions depending on the data. The model should be used only for educational purposes, not for real-world decisions or applications.

## Caveats and Recommendations
The model's training data is limited and is unlikely to perform well on real-world data. The model can be improved with tuning, more testing, and a larger dataset.