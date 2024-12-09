import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import StackingClassifier

print("")

data = pd.DataFrame()

# Import the local data for the plays
for year in range(2020, 2024):
    filename = f"play_by_play_{year}_new.csv"
    if os.path.exists(filename):
        year_data = pd.read_csv(filename)
        data = pd.concat([data, year_data], ignore_index=True)
    else:
        print(f"{filename} not found.")

data.fillna(0, inplace=True)

                                                                                                                            # Basic Features
data['red_zone'] = (data['yardline_100'] <= 20).astype(int)                                                                 # Is this within the 20 yard line (red zone)?
data['is_home_team'] = (data['home_team'] == data['posteam']).astype(int)                                                   # Is this the home team?
data['is_late_season'] = (data['week'] >= 10).astype(int)                                                                   # Is this late in the season?
data['is_shotgun'] = data['shotgun'].astype(int)                                                                            # Is the team lined up in shotgun formation?
data['is_no_huddle'] = data['no_huddle'].astype(int)                                                                        # Huddle prior to the play?
data['near_end_of_half'] = (data['half_seconds_remaining'] < 240).astype(int)                                               # Is it within 4 minutes of the end of the half?
data['near_end_of_game'] = (data['game_seconds_remaining'] < 240).astype(int)                                               # Is it within 4 minutes of the end of the g?

                                                                                                                            # Advanced Features
data['score_diff'] = data.apply(                                                                                            # - This one can't even be used cause it doesn't help but it's useful for the others.
    lambda row: row['home_score'] - row['away_score'] if row['is_home_team'] == 1 
    else row['away_score'] - row['home_score'], axis=1
)
data['late_cozy_offense'] = ((data['half_seconds_remaining'] <= 240) & (data['score_diff'] > 0)).astype(int)                # - Late in the game, team is in the lead (run usually more likely).
data['late_desperate_offense'] = ((data['half_seconds_remaining'] <= 240) & (data['score_diff'] < 0)).astype(int)           # - Late in the game, team is behind (pass usually more likely).
data['late_super_desperate_offense'] = ((data['half_seconds_remaining'] <= 240) & (data['score_diff'] < -8)).astype(int)    # - Late in the game, team is very behind (pass usually more likely).
data['rolling_avg_yards'] = data['yards_gained'].rolling(window=3, min_periods=1).mean()                                    # - Attempt at a momentum feature.
data['previous_play_run'] = (data['play_type'].shift(1) == 'run').astype(int)                                               # - Was the previous play a run?
data['previous_play_pass'] = (data['play_type'].shift(1) == 'pass').astype(int)                                             # - Was the previous play a pass?
data['past_conversion_success'] = (data['third_down_converted'].shift(1).astype(bool)                                       # - Recent third/fourth down conversion success?
                                 | data['fourth_down_converted'].shift(1).astype(bool)).astype(int)

# The features to be included in the models.
selected_features = [
    'down',
    'ydstogo',
    'red_zone',
    'is_home_team',
    'is_late_season',
    'is_shotgun',
    'is_no_huddle',
    'near_end_of_half',
    'near_end_of_game',
    'late_cozy_offense',
    'late_desperate_offense',
    'late_super_desperate_offense',
    'rolling_avg_yards',
    'previous_play_run',
    'previous_play_pass',
    'past_conversion_success',
    ]

# Play type is our target variable; filter out unwanted plays like punts and kickoffs
filtered_data = data[data['play_type'].isin(['pass', 'run'])]  # Limit to pass/run plays
filtered_data = filtered_data[filtered_data['down'] != 0] # For some reason some "0th down" plays were still left over

X = filtered_data[selected_features]
y = filtered_data['play_type'].apply(lambda x: 1 if x == 'pass' else 0)  # Encode 'pass' as 1, 'run' as 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating Decision Tree model. Depth of 9 turned out to be optimal.
dt_model = DecisionTreeClassifier(max_depth=9, random_state=42)  # Limit depth to avoid overfitting
dt_model.fit(X_train, y_train)

# Creating Logistic Regression model. Depth of 9 turned out to be optimal.
lr_model = LogisticRegression(max_iter=1000, random_state=42)  # Increased iterations for convergence
lr_model.fit(X_train, y_train)

stacked_model = StackingClassifier(
    estimators=[
        ('decision_tree', dt_model),
        ('logistic_regression', lr_model)
    ],
    final_estimator=LogisticRegression(random_state=42)
)

stacked_model.fit(X_train, y_train)

# Making predictions based on decision tree model.
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)
dt_f1 = f1_score(y_test, dt_predictions)

# Making predictions based on regression model.
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

# Making prediction based on stacked model.
stacked_predictions = stacked_model.predict(X_test)
accuracy = accuracy_score(y_test, stacked_predictions)
precision = precision_score(y_test, stacked_predictions)
recall = recall_score(y_test, stacked_predictions)
f1 = f1_score(y_test, stacked_predictions)

print("Decision Tree Performance:")
print(f"Accuracy: {dt_accuracy}, Precision: {dt_precision}, Recall: {dt_recall}, F1 Score: {dt_f1}")

print("\nLogistic Regression Performance:")
print(f"Accuracy: {lr_accuracy}, Precision: {lr_precision}, Recall: {lr_recall}, F1 Score: {lr_f1}")

print("\nStacking Performance:")
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

# Preparing results for printing.
stacked_results = X_test.copy()
stacked_results['actual_play'] = y_test.map({1: 'pass', 0: 'run'})
stacked_results['predicted_play'] = stacked_predictions
stacked_results['predicted_play'] = stacked_results['predicted_play'].map({1: 'pass', 0: 'run'})
stacked_results['yardline_100'] = filtered_data.loc[X_test.index, 'yardline_100'].astype(int)
stacked_results['down'] = stacked_results['down'].astype(int)

# Features included printout.
'''
print("")
print("Features: ")
print(stacked_results.columns)
print("")
'''

# Labeling if the actual play matches the predicted play, i.e. correct prediction.
stacked_results['match'] = stacked_results.apply(
    lambda row: '✔' if row['actual_play'] == row['predicted_play'] else '✘', axis=1
)

# Columns to be displayed in printout.
display_cols = [
    'down', 
    'ydstogo', 
    'yardline_100',
    'near_end_of_half',
    'predicted_play',
    'actual_play',
    'match',
]

# Printing out prediction results
print("")
print(stacked_results[display_cols].head(10))

# Print single result for analysis
'''
r = stacked_results.loc[46344]
print(r)
print("")
'''

# Debugging
'''
mis_classified = stacked_results[stacked_results['actual_play'] != stacked_results['predicted_play']]
print("Misidentified Plays")
print(mis_classified.groupby('down').size()) 
print(mis_classified.head())
print("")
print(mis_classified.describe())
print("")
print("Average Yards to Go (Misclassified vs. All Plays):")
print("Misclassified:", mis_classified['ydstogo'].mean())
print("All Plays:", stacked_results['ydstogo'].mean())
print("")
print("Average Yardline Position (Misclassified vs. All Plays):")
print("Misclassified:", mis_classified['yardline_100'].mean())
print("All Plays:", stacked_results['yardline_100'].mean())

correctly_classified = stacked_results[stacked_results['actual_play'] == stacked_results['predicted_play']]
print("Correctly Identified Plays")
print(correctly_classified.groupby('down').size()) 
print("")

print("Total Plays In Dataset")
print(filtered_data.groupby('down').size()) 
print(filtered_data.groupby('play_type').size()) 

print("")
total_classified = len(mis_classified) + len(correctly_classified)
print(f"Total plays classified: {total_classified}, Total plays in X_test: {len(X_test)}")
print(f"Unique values in y_test: {y_test.unique()}")
print(f"Unique values in stacked_results['actual_play']: {stacked_results['actual_play'].unique()}")
print(f"Unique values in stacked_results['predicted_play']: {stacked_results['predicted_play'].unique()}")
print(f"Length of X_test: {len(X_test)}, Length of y_test: {len(y_test)}")
print(f"Rows in y_test: {len(y_test)}, Rows in stacked_predictions: {len(stacked_predictions)}")
print(f"Length of stacked_results: {len(stacked_results)}")
'''

# Trying to predict the location/length/gap of plays too.
print("")
