import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools

pd.options.mode.copy_on_write = True


def parse_date(date):
    if pd.isnull(date) or date == "":
        return None
    else:
        date = str(date)
        return dt.strptime(date, "%d/%m/%y").date()


def get_goals_scored(playing_stat):
    teams = {}
    for i in playing_stat['HomeTeam'].unique():
        print("check {} \n".format(i))
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
        print("check {} \n".format(i))
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

    #     print("check line 87")
    #     print(playing_stat.shape,len(HTGS))

    playing_stat['HTGS'] = HTGS
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC

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
        print("check {} \n".format(i))
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
    # TODO: matchdays smaller then j
    print(matchdays)
    for i in range(len(playing_stat)):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cuml_pts.loc[ht][j])
        ATP.append(cuml_pts.loc[at][j])

        if ((i + 1) % 10) == 0:
            if j > matchdays:
                j = matchdays
            else:
                j = j + 1

    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
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
    for num in range(1, 5):
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
        playing_stat['HM' + str(num)] = h[:playing_stat.shape[0]]
        playing_stat['AM' + str(num)] = a[:playing_stat.shape[0]]

    return playing_stat


if __name__ == "__main__":

    loc = "data\\"
    file_names = [f"pl{year:02d}-{year + 1:02d}.csv" for year in range(3, 24)]

    dataframes = [pd.read_csv(loc + file_name) for file_name in file_names]

    for df in dataframes:
        df["Date"] = df["Date"].apply(parse_date)

    columns_for_playing_stats = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    playing_stats = [get_gss(df[columns_for_playing_stats]) for df in dataframes]
    playing_stats = [get_agg_points(df) for df in playing_stats]
    playing_stats = [add_form(df) for df in playing_stats]
    print(playing_stats[12].head())

    playing_stats[12]
