from time import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import pandas as pd
from model_evaluation import split_match_data, get_model_metrics, get_scores, plot_scores
from scikeras.wrappers import KerasClassifier
from models import get_LSTM
models = [
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression(max_iter=1000),
    SVC(),
    LinearSVC(),
    NuSVC(),
    KNeighborsClassifier(n_neighbors=22),
    KerasClassifier(
        get_LSTM,
        loss="sparse_categorical_crossentropy",
    )
]

# list_of_features = [
#    'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
#    'HM1', 'AM1', 'HM2', 'AM2', 'HM3', 'AM3', 'HM4', 'AM4', 'HM5', 'AM5', 'MW', 'HomeTeamLP',
#    'AwayTeamLP', 'HTFormPtsStr', 'ATFormPtsStr', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
#    'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5',
#    'ATLossStreak3', 'ATLossStreak5', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'DiffLP'
# ]

list_of_features = [
    'HTP', 'ATP',
    'HM1', 'AM1', 'HM2', 'AM2', 'HM3', 'AM3', 'HM4', 'AM4', 'HM5', 'AM5', 'HomeTeamLP',
    'AwayTeamLP', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
    'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5',
    'ATLossStreak3', 'ATLossStreak5', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'DiffLP'
]

target_variable = 'FTR'


def main():
    start = time()
    match_data = pd.read_csv("data/all_stats.csv")
    X_train, X_test, y_train, y_test = split_match_data(match_data, list_of_features, target_variable)
    metrics_df = get_model_metrics(models, X_train, X_test, y_train, y_test)
    get_scores(metrics_df)
    plot_scores(metrics_df)
    end = time()
    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    main()
