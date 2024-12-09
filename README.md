# NFL Play Prediction Model

This project predicts offensive play-calling in the NFL (pass vs. run) using pre-snap data. The dataset includes NFL play-by-play data from 2024 to 2024, downloaded in `.csv` format from
the [nflverse GitHub collection](https://github.com/nflverse/nflverse-data/releases/tag/pbp). The data was cleaned using `clean.py` to produce the `play_by_play_(year)_new.csv` files containing only the needed columns for training.

The machine learning workflow is implemented in `MLProject.ipynb`/`MLProject.py`, where:
- Additional features are engineered, including both basic and more complex features.  
- Decision Tree, Logistic Regression, and Stacked models are trained to predict the play type.  
- The models are evaluated on the following metrics: accuracy, precision, recall, and F1 score.  
- Predictions are displayed, identifying whether or not the predicted play-call matched the actual result.  

Additionally, the notebook provides functionality for singular custom predictions, allowing users to modify pre-snap features and observe the corresponding model predictions.

### Data Bounds
Only data from 2020–2024 is included in this repository to focus on predicting current play-calling patterns. Including earlier data tended to negatively influence accuracy results. However, you can adjust these bounds by downloading and cleaning the appropriate `.csv` files from the [nflverse GitHub collection](https://github.com/nflverse/nflverse-data/releases/tag/pbp), and altering the `for year in range(2020, 2024):` line in either file.
