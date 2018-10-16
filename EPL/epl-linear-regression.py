import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

def season_text_to_season_index(season_text):
    season_start_text = season_text[4:]
    season_start = int(season_start_text) - 1992

    return season_start

def preprocess_features(epl_results_dataframe):
    selected_features = epl_results_dataframe[
        ["HomeTeam",
        "AwayTeam"]
    ]

    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["Season"] = epl_results_dataframe["Season"].apply(season_text_to_season_index)

    return processed_features

def preprocess_targets(epl_results_dataframe):
    selected_targets = epl_results_dataframe[
        ["FTHG",
        "FTAG"]
    ]

    return selected_targets

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(teams_vocabulary_list):
    feature_columns = [
        tf.feature_column.categorical_column_with_vocabulary_list("HomeTeam", teams_vocabulary_list),
        tf.feature_column.categorical_column_with_vocabulary_list("AwayTeam", teams_vocabulary_list),
        tf.feature_column.numeric_column("Season", dtype=tf.int32)
    ]
    
    return feature_columns

def train_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear regressor object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      label_dimension=2,
      optimizer=my_optimizer
  )
  
  # 1. Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, training_targets, batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets, num_epochs=1, shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets, num_epochs=1, shuffle=False)
  
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_hg_rmse = []
  training_ag_rmse = []
  validation_hg_rmse = []
  validation_ag_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period,
    )
    # 2. Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_hg_predictions = np.array([item['predictions'][0] for item in training_predictions])
    # TODO: this is coming out empty
    training_ag_predictions = np.array([item['predictions'][1] for item in training_predictions])

    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_hg_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    # TODO: this is coming out empty
    validation_ag_predictions = np.array([item['predictions'][1] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_hg_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_hg_predictions, training_targets["FTHG"]))
    validation_hg_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_hg_predictions, validation_targets["FTHG"]))
    # training_ag_root_mean_squared_error = math.sqrt(
    #     metrics.mean_squared_error(training_ag_predictions, training_targets["FTAG"]))
    # validation_ag_root_mean_squared_error = math.sqrt(
    #     metrics.mean_squared_error(validation_ag_predictions, validation_targets["FTAG"]))
    # Occasionally print the current loss.
    print("  period %02d : HG: %0.2f" % (period, training_hg_root_mean_squared_error))
    #print("  period %02d : AG: %0.2f" % (period, training_ag_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_hg_rmse.append(training_hg_root_mean_squared_error)
    validation_hg_rmse.append(validation_hg_root_mean_squared_error)
    #training_ag_rmse.append(training_ag_root_mean_squared_error)
    #validation_ag_rmse.append(validation_ag_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_hg_rmse, label="training (HG)")
  plt.plot(validation_hg_rmse, label="validation (HG)")
  plt.plot(training_ag_rmse, label="training (AG)")
  plt.plot(validation_ag_rmse, label="validation (AG)")
  plt.legend()

  return linear_regressor

epl_results_dataframe = pd.read_csv("EPL_Set.csv", sep=",")

epl_results_dataframe = epl_results_dataframe.reindex(
    np.random.permutation(epl_results_dataframe.index))

num_records = epl_results_dataframe.count()[0]
num_training_examples = int(num_records * 0.75)
num_validation_examples = num_records - num_training_examples

training_examples = preprocess_features(epl_results_dataframe.head(num_training_examples))
training_targets = preprocess_targets(epl_results_dataframe.head(num_training_examples))
validation_examples = preprocess_features(epl_results_dataframe.tail(num_validation_examples))
validation_targets = preprocess_targets(epl_results_dataframe.tail(num_validation_examples))

teams_vocabulary_list = set(epl_results_dataframe["HomeTeam"])
feature_columns=construct_feature_columns(teams_vocabulary_list)

linear_regressor = train_model(
    learning_rate=0.0001,
    steps=1000,
    batch_size=20,
    feature_columns=feature_columns,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

plt.show()