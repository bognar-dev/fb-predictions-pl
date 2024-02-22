import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt


def split_match_data(match_data, list_of_features: list[str], target_variable: str, test_size=0.3, random_state=42):
    # list_of_features = ['home_team_api_id', 'away_team_api_id', 'league_id', 'home_team_goal', 'away_team_goal',
    #                     'overall_rating_home', 'overall_rating_away', 'overall_rating_difference',
    #                     ]
    # target_variable = 'home_status'
    X = match_data[list_of_features]
    y = match_data[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def get_model_metrics(models, X_train, X_test, y_train, y_test):
    modelMetrics = []
    for model in models:
        model_name = model.__class__.__name__
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_score = model.score(X_test, y_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        y_true = y_test
        y_pred = predictions
        plot_confusionMatrix(y_true, y_pred)
        modelMetrics.append(
            {'Model': model_name, 'Score': model_score, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall,
             'F1-score': f1, 'Confusion Matrix': conf_matrix})

    df = pd.DataFrame(modelMetrics)

    return df


def get_scores(modelMetricsDf):
    modelMetrics = modelMetricsDf.to_dict('records')
    for modelmetric in modelMetrics:
        print(f"{modelmetric['Model']} Score is {str(modelmetric['Score'])[:4]} ")


def plot_scores(df):
    plt.figure(figsize=(10, 5))
    plt.bar(df['Model'], df['Accuracy'], label='Accuracy')
    plt.plot(df['Model'], df['Precision'], label='Precision', color='orange')
    plt.plot(df['Model'], df['Recall'], label='Recall', color='green')
    plt.plot(df['Model'], df['F1-score'], label='F1-score', color='red')
    plt.plot(df['Model'], df['Score'], label='Score', color='purple')
    plt.title('Model Metrics')
    plt.xlabel('Model')
    plt.ylabel('Metrics')
    plt.legend()
    plt.show()


def plot_confusionMatrix(y_true, y_pred):
    cn = confusion_matrix(y_true=y_true, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cn, annot=True, linewidths=1.5)
    plt.show()
    return cn
