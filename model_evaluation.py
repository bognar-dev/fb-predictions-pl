import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import textwrap

from sklearn.preprocessing import LabelEncoder


def split_match_data(match_data, list_of_features: list[str], target_variable: str, test_size=0.3, random_state=42):

    X = match_data[list_of_features]
    y = match_data[target_variable]
    le = LabelEncoder()
    y = le.fit_transform(y)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def get_model_metrics(models, X_train, X_test, y_train, y_test):
    batch_size = 64
    model_metrics = {}
    for model in models:
        model_name = model.__class__.__name__
        if model_name == 'KerasClassifier':
            model.fit(
                X_train, y_train, validation_data=(X_test, y_test), batch_size=batch_size, epochs=30
            )
        else:
            model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_score = model.score(X_test, y_test)
        accuracy = accuracy_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted',zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        y_true = y_test
        y_pred = predictions
        # plot_confusionMatrix(y_true, y_pred)

        model_metrics[model_name] = {
            'Score': model_score,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }

    return model_metrics


def get_scores(modelMetrics):
    for model_name, metrics in modelMetrics.items():
        print(f"{model_name} Score is {str(metrics['Score'])[:4]}")


def wrap_labels(ax, width, break_long_words=True):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                                    break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


def plot_scores(model_metrics):
    # Define the label locations and the width of the bars
    x = np.arange(len(model_metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a color palette
    colours = sns.color_palette('pastel')

    # Specify the order of metrics
    metrics_order = ['Score', 'Recall', 'Accuracy', 'F1-score', 'Precision']

    # Create a bar for each metric in the specified order
    for metric_id, metric in enumerate(metrics_order):
        values = [metrics[metric] for metrics in model_metrics.values()]
        rects = ax.bar(x + width * metric_id, values, width, label=metric, color=colours[metric_id])

    # find min and max values for y axis and limit the y axis
    min_y = min([min([metrics[metric] for metrics in model_metrics.values()]) for metric in metrics_order])
    max_y = max([max([metrics[metric] for metrics in model_metrics.values()]) for metric in metrics_order])
    ax.set_ylim([min_y - 0.05, max_y + 0.05])

    # Add labels, title, legend, etc.
    ax.set_ylabel('Metrics')
    ax.set_title('Model Metrics by Model')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_metrics.keys())
    ax.legend()

    wrap_labels(ax, 10)

    plt.show()


def plot_confusionMatrix(y_true, y_pred):
    cn = confusion_matrix(y_true=y_true, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cn, annot=True, linewidths=1.5)
    plt.show()
    return cn
