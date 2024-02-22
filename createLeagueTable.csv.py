import csv


def create_league_csv(input_file, output_file):
    team_data = {}
    with open(input_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            team = row[2]
            season = row[0]
            pos = row[1]
            if team not in team_data:
                team_data[team] = {}
            # Extracting only the last two digits of the season
            short_season = season[-2:]
            team_data[team][short_season] = pos

    # Determine all unique seasons
    seasons = sorted(set(season for team in team_data.values() for season in team.keys()))

    # Write the league table to output file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        header_row = ['Team'] + seasons
        writer.writerow(header_row)
        # Write each team's data
        for team, positions in team_data.items():
            row_data = [team]
            for season in seasons:
                row_data.append(positions.get(season, ''))
            writer.writerow(row_data)


create_league_csv('./data/input.csv', 'data/plStandings.csv')
