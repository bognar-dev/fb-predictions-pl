import pandas as pd

loc = "data\\"
file_names = [f"pl{year:02d}-{year + 1:02d}.csv" for year in range(3, 24)]
standings = pd.read_csv(loc + "plStandings.csv")
print(standings.index)
standings.set_index(['Team'], inplace=True)
standings = standings.fillna(18)
dataframes = [pd.read_csv(loc + file_name) for file_name in file_names]

# Create a dictionary from the pl_seasons DataFrames
pl_seasons = pd.concat(dataframes, ignore_index=True)
#get all unique team names
teams = pl_seasons.HomeTeam.unique()

team_name_dict = pd.Series(pl_seasons.HomeTeam.values, index=pl_seasons.HomeTeam).to_dict()

# Replace the team names in the pl_standings DataFrame
standings['Team'] = standings['Team'].map(team_name_dict)

# Save the updated DataFrame to a new CSV file
standings.to_csv(loc + 'updated_pl_standings.csv', index=True, index_label='Team')