import pandas as pd
import os

selected_columns = [ 
'play_id', 'game_id', 'home_team', 'away_team', 'posteam', 'defteam',
'season_type', 'week', 'down', 'ydstogo', 'yardline_100', 'play_type',
'yards_gained', 'epa', 'success', 'touchdown', 'first_down', 'pass',
'rush', 'field_goal_attempt', 'air_yards', 'comp_air_epa', 'qb_epa',
'third_down_converted', 'fourth_down_converted', 'no_huddle', 'shotgun',
'qb_dropback', 'qb_kneel', 'qb_spike', 'two_point_attempt', 'interception',
'tackled_for_loss', 'sack', 'complete_pass', 'fumble', 'series_result',
'series_success', 'fixed_drive_result', 'run_location','run_gap','pass_length',
'pass_location','qb_dropback','qb_scramble','yards_after_catch','qtr',
'half_seconds_remaining', 'game_seconds_remaining','away_score','home_score',
'home_timeouts_remaining', 'away_timeouts_remaining'
]

for year in range(1999, 2025):
    filename = f"play_by_play_{year}.csv"
    new_filename = f"play_by_play_{year}_new.csv"
    
    if os.path.exists(filename):

        df = pd.read_csv(filename)
        df_filtered = df[selected_columns]
        df_filtered.to_csv(new_filename, index=False)
        print(f"Processed and saved {new_filename}")
    else:
        print(f"{filename} not found.")
