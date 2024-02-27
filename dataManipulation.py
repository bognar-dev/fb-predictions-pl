import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

pd.options.mode.copy_on_write = True


def parse_date(date, fmt="%d/%m/%y"):
    if pd.isnull(date) or date == "":
        return None
    else:
        date = str(date)
        return dt.strptime(date, fmt).date()


def get_goals_scored(playing_stat):
    teams = {}
    for i in playing_stat['HomeTeam'].unique():
        teams[i] = []

    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)

    min_matchdays = min(len(v) for v in teams.values())
    if min_matchdays != 38:
        # cut of the extra matchdays teams have played more than the others
        for key in teams.keys():
            teams[key] = teams[key][:min_matchdays]
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, min_matchdays + 1)]).T
    GoalsConceded[0] = 0
    for i in range(2, min_matchdays + 1):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i - 1]
    return GoalsConceded


def get_goals_conceded(playing_stat):
    teams = {}
    for i in playing_stat['HomeTeam'].unique():
        teams[i] = []

    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)

    min_matchdays = min(len(v) for v in teams.values())
    if min_matchdays != 38:
        # cut of the extra matchdays teams have played more than the others
        for key in teams.keys():
            teams[key] = teams[key][:min_matchdays]
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, min_matchdays + 1)]).T
    GoalsConceded[0] = 0
    for i in range(2, min_matchdays + 1):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i - 1]
    return GoalsConceded


def get_gss(playing_stat):
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)

    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1

    HTGS = pd.Series(HTGS, name='HTGS')
    ATGS = pd.Series(ATGS, name='ATGS')
    HTGC = pd.Series(HTGC, name='HTGC')
    ATGC = pd.Series(ATGC, name='ATGC')

    playing_stat = pd.concat([playing_stat, HTGS], axis=1)
    playing_stat = pd.concat([playing_stat, ATGS], axis=1)
    playing_stat = pd.concat([playing_stat, HTGC], axis=1)
    playing_stat = pd.concat([playing_stat, ATGC], axis=1)

    return playing_stat


def get_points(result):
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0


def get_matches(playing_stat):
    teams = {}
    for i in playing_stat['HomeTeam'].unique():
        teams[i] = []

    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')

    min_matchdays = min(len(v) for v in teams.values())
    if min_matchdays != 38:
        # cut of the extra matchdays teams have played more than the others
        for key in teams.keys():
            teams[key] = teams[key][:min_matchdays]
    return pd.DataFrame(data=teams, index=[i for i in range(1, min_matchdays + 1)]).T


def get_cuml_points(matches):
    matches_points = matches.map(get_points)
    for i in range(2, len(matches_points.columns) + 1):
        matches_points.iloc[:, i - 1] = matches_points.iloc[:, i - 1] + matches_points.iloc[:, i - 2]

    return matches_points


def get_agg_points(playing_stat):
    matches = get_matches(playing_stat)
    cuml_pts = get_cuml_points(matches)
    HTP = []
    ATP = []
    j = 1
    matchdays = len(cuml_pts.columns)
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cuml_pts.loc[ht][j])
        ATP.append(cuml_pts.loc[at][j])
        # Ensure j is within the range of cuml_pts columns
        j = min(j, matchdays - 1)
        if ((i + 1) % 10) == 0:
            j = j + 1

    HTP = pd.Series(HTP, name='HTP')
    ATP = pd.Series(ATP, name='ATP')
    playing_stat = pd.concat([playing_stat, HTP], axis=1)
    playing_stat = pd.concat([playing_stat, ATP], axis=1)

    return playing_stat


def get_form(playing_stat, num):
    form = get_matches(playing_stat)
    form_final = form.copy()
    for i in range(num, len(form)):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i - j]
            j += 1
    return form_final


def add_form(playing_stat):
    new_columns = {}
    for num in range(1, 6):
        form = get_form(playing_stat, num)
        h = ['N' for i in range(num * 10)]  # since form is not available for n MW (n*10)
        a = ['N' for i in range(num * 10)]

        j = num
        for i in range((num * 10), len(playing_stat)):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam

            past = form.loc[ht][j]  # get past n results
            if len(past) >= num:
                h.append(past[num - 1])  # 0 index is most recent
            else:
                h.append('N')

            past = form.loc[at][j]  # get past n results.
            if len(past) >= num:
                a.append(past[num - 1])  # 0 index is most recent
            else:
                a.append('N')

            if ((i + 1) % 10) == 0:
                j = j + 1
        new_columns['HM' + str(num)] = h[:playing_stat.shape[0]]
        new_columns['AM' + str(num)] = a[:playing_stat.shape[0]]

    playing_stat = pd.concat([playing_stat, pd.DataFrame(new_columns)], axis=1)

    return playing_stat


def get_mw(playing_stat):
    matches = get_matches(playing_stat)
    cuml_pts = get_cuml_points(matches)
    j = 1
    matchdays = len(cuml_pts.columns)
    MatchWeek = []
    for i in range(len(playing_stat)):
        MatchWeek.append(j)
        j = min(j, matchdays - 1)
        if ((i + 1) % 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat


def get_form_points(string):
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum


def calculate_form_points(playing_stat):
    playing_stat['HTFormPtsStr'] = playing_stat['HM1'] + playing_stat['HM2'] + playing_stat['HM3'] + playing_stat[
        'HM4'] + playing_stat['HM5']
    playing_stat['ATFormPtsStr'] = playing_stat['AM1'] + playing_stat['AM2'] + playing_stat['AM3'] + playing_stat[
        'AM4'] + playing_stat['AM5']

    playing_stat['HTFormPts'] = playing_stat['HTFormPtsStr'].apply(get_form_points)
    playing_stat['ATFormPts'] = playing_stat['ATFormPtsStr'].apply(get_form_points)

    return playing_stat


def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0


def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0


def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0


def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0


def calculate_streaks(playing_stat):
    playing_stat['HTWinStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ws)
    playing_stat['HTWinStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ws)
    playing_stat['HTLossStreak3'] = playing_stat['HTFormPtsStr'].apply(get_3game_ls)
    playing_stat['HTLossStreak5'] = playing_stat['HTFormPtsStr'].apply(get_5game_ls)

    playing_stat['ATWinStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ws)
    playing_stat['ATWinStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ws)
    playing_stat['ATLossStreak3'] = playing_stat['ATFormPtsStr'].apply(get_3game_ls)
    playing_stat['ATLossStreak5'] = playing_stat['ATFormPtsStr'].apply(get_5game_ls)

    return playing_stat


def calculate_differences(playing_stat):
    # Get Goal Difference
    playing_stat['HTGD'] = playing_stat['HTGS'] - playing_stat['HTGC']
    playing_stat['ATGD'] = playing_stat['ATGS'] - playing_stat['ATGC']

    # Diff in points
    playing_stat['DiffPts'] = playing_stat['HTP'] - playing_stat['ATP']
    playing_stat['DiffFormPts'] = playing_stat['HTFormPts'] - playing_stat['ATFormPts']

    # Diff in last year positions
    playing_stat['DiffLP'] = playing_stat['HomeTeamLP'] - playing_stat['AwayTeamLP']

    return playing_stat


def get_last_positions(playing_stat, Standings):
    year = str(parse_date(playing_stat['Date'].min(), "%Y-%m-%d").year - 1)[-2:]
    HomeTeamLP = []
    AwayTeamLP = []
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        # Check if the team's position is NaN (i.e. team was not in the league last year) and assign 21 if promoted
        HomeTeamLP.append(Standings.loc[ht][year] if pd.notna(Standings.loc[ht][year]) else 21)
        AwayTeamLP.append(Standings.loc[at][year] if pd.notna(Standings.loc[at][year]) else 21)
    HomeTeamLP = pd.Series(HomeTeamLP, name='HomeTeamLP')
    AwayTeamLP = pd.Series(AwayTeamLP, name='AwayTeamLP')
    playing_stat = pd.concat([playing_stat, HomeTeamLP], axis=1)
    playing_stat = pd.concat([playing_stat, AwayTeamLP], axis=1)
    return playing_stat


def form_one_hot_encoding(playing_stat):
    mapping = {'N': -1, 'L': 0, 'D': 1, 'W': 3}

    for col in ['HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']:
        playing_stat[col] = playing_stat[col].map(mapping)

    # Assuming df is your DataFrame and 'column' is the column with the 'LDWNN' like strings
    # Convert the string into a list of characters
    playing_stat['ATFormPtsStr'] = playing_stat['ATFormPtsStr'].apply(list)
    playing_stat['HTFormPtsStr'] = playing_stat['HTFormPtsStr'].apply(list)
    # Apply LabelEncoder to each list of characters in the column
    le = LabelEncoder()
    playing_stat['ATFormPtsStr'] = playing_stat['ATFormPtsStr'].apply(le.fit_transform)
    playing_stat['HTFormPtsStr'] = playing_stat['HTFormPtsStr'].apply(le.fit_transform)

    return playing_stat


def scale_features(playing_stat):
    cols = ['HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'HTP', 'ATP']
    playing_stat.MW = playing_stat.MW.astype(float)

    for col in cols:
        playing_stat[col] = playing_stat[col] / playing_stat.MW

    playing_stat.HTGS = playing_stat.HTGS / playing_stat.HTGS.max()
    playing_stat.ATGS = playing_stat.ATGS / playing_stat.ATGS.max()
    playing_stat.HTGC = playing_stat.HTGC / playing_stat.HTGC.max()
    playing_stat.ATGC = playing_stat.ATGC / playing_stat.ATGC.max()

    return playing_stat


def one_hot_encode_matches(playing_stat):
    # Create a LabelEncoder
    le = LabelEncoder()

    # Apply the LabelEncoder to the 'HM1' to 'AM5' columns
    for col in ['HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']:
        playing_stat[col] = le.fit_transform(playing_stat[col])

    # One-hot encode the 'FTR' column
    onehotencoder = OneHotEncoder()
    final = onehotencoder.fit_transform(playing_stat['FTR'].values.reshape(-1,1)).toarray()

    # Add the one-hot encoded columns to the dataframe
    playing_stat.loc[:,"final1"] = final[:,0]
    playing_stat.loc[:,"final2"] = final[:,1]

    return playing_stat


if __name__ == "__main__":

    loc = "data\\"
    file_names = [f"pl{year:02d}-{year + 1:02d}.csv" for year in range(3, 24)]
    standings = pd.read_csv(loc + "plStandings.csv")
    print(standings.index)
    standings.set_index(['Team'], inplace=True)
    dataframes = [pd.read_csv(loc + file_name) for file_name in file_names]
    for df in dataframes:
        df["Date"] = df["Date"].apply(parse_date)
    columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    dataframes = [df[columns_req] for df in dataframes]
    playing_stats = [get_gss(df) for df in
                     tqdm(dataframes, desc="Processing goal statistics", ncols=100)]
    playing_stats = [get_agg_points(df) for df in tqdm(playing_stats, desc="Aggregating points", ncols=100)]
    playing_stats = [add_form(df) for df in tqdm(playing_stats, desc="Adding recent form", ncols=100)]
    playing_stats = [get_mw(df) for df in tqdm(playing_stats, desc="Adding matchweeks", ncols=100)]
    playing_stats = [get_last_positions(df, standings) for df in
                     tqdm(playing_stats, desc=f"Adding last year positions", ncols=100)]
    playing_stats = [calculate_form_points(df) for df in tqdm(playing_stats, desc="Calculating form points", ncols=100)]
    playing_stats = [calculate_streaks(df) for df in tqdm(playing_stats, desc="Calculating streaks", ncols=100)]
    playing_stats = [calculate_differences(df) for df in tqdm(playing_stats, desc="Calculating differences", ncols=100)]
    playing_stats = [form_one_hot_encoding(df) for df in tqdm(playing_stats, desc="One hot encoding form", ncols=100)]
    playing_stats = [scale_features(df) for df in tqdm(playing_stats, desc="Scaling features", ncols=100)]
    playing_stats = [one_hot_encode_matches(df) for df in tqdm(playing_stats, desc="One hot encoding matches", ncols=100)]
    # save all playing stats in one csv
    playing_stats = pd.concat(playing_stats)
    playing_stats.to_csv("data\\all_stats.csv", index=False)
    print("All stats saved to all_stats.csv")
