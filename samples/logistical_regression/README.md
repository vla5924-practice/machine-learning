# Logistical regression sample

Runs logistical regression model training and then test input data classification based on pre-trained model.

## Usage

```sh
./logistical_regression_sample PATH_TO_TRAINING_DATASET [PATH_TO_TESTING_DATASET]
```

Command line arguments:
* `PATH_TO_TRAINING_DATASET` - path to CSV file with dataset to train with
* `PATH_TO_TESTING_DATASET` _(optional)_ - path to CSV file with dataset to test out

Working modes:
* **Output mode.** This mode is selected if `PATH_TO_TESTING_DATASET` parameter is specified. It parses both files, executes model training, then outputs classification results for each line in CSV format directly to console.
* **Interactive mode.** This mode is selected if `PATH_TO_TESTING_DATASET` parameter is _not_ specified. It parses dataset file, executes model training, then infinitely asks user to enter feature values as input to be classified and outputs classification results.
