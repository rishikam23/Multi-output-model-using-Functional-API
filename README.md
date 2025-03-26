# Multi-Output Model using Functional API

## Overview
This project implements a multi-output regression model using TensorFlow's Functional API. The model predicts two target variables (`Y1` and `Y2`) from a given dataset. The dataset is preprocessed, normalized, and split into training and test sets. The model consists of dense layers with ReLU activations and separate output layers for each target variable.

## Dataset
The dataset used is the **Energy Efficiency Dataset** from the UCI Machine Learning Repository:
[Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx)

### Features
- The dataset contains various building parameters such as relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area, and glazing area distribution.
- The targets (`Y1` and `Y2`) represent **heating load** and **cooling load**, respectively.

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install tensorflow numpy matplotlib pandas scikit-learn openpyxl
```

## Implementation Details
1. **Data Preprocessing**:
   - Load the dataset using Pandas.
   - Shuffle and split into 80% training and 20% testing data.
   - Normalize the input features using mean and standard deviation.
   - Extract target values (`Y1` and `Y2`).

2. **Model Architecture**:
   - Input layer with shape `(number_of_features,)`.
   - Two dense layers (128 neurons each, ReLU activation).
   - Separate branches for each output:
     - `Y1_output`: Directly connected to the second dense layer.
     - `Y2_output`: Connected via an additional dense layer (64 neurons, ReLU activation).

3. **Compilation & Training**:
   - Uses **Stochastic Gradient Descent (SGD)** optimizer with a learning rate of `0.001`.
   - Mean Squared Error (`mse`) as the loss function for both outputs.
   - Root Mean Squared Error (`rmse`) as an evaluation metric.
   - Trained for **500 epochs** with a batch size of **10**.

4. **Evaluation & Visualization**:
   - Evaluates the model on test data and prints loss and RMSE values.
   - Plots:
     - **True vs. Predicted Values** for both `Y1` and `Y2`.
     - **RMSE trends** over training epochs.

## Usage
Run the script using:
```bash
python multi_output_model_using_functional_api.py
```

## Model Summary
The model structure:
```plaintext
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, num_features)]    0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)              X         
_________________________________________________________________
dense_2 (Dense)              (None, 128)              X         
_________________________________________________________________
y1_output (Dense)            (None, 1)                X         
_________________________________________________________________
dense_3 (Dense)              (None, 64)               X         
_________________________________________________________________
y2_output (Dense)            (None, 1)                X         
=================================================================
```

## Results
After training, the model outputs evaluation metrics such as:
```plaintext
Loss = X, Y1_loss = X, Y1_rmse = X, Y2_loss = X, Y2_rmse = X
```

## Future Improvements
- Experiment with different architectures (e.g., additional layers, dropout, batch normalization).
- Use advanced optimizers like **Adam**.
- Hyperparameter tuning with `KerasTuner`.
- Implement real-world deployment using **Flask or FastAPI**.

## License
This project is open-source and free to use.

---
Feel free to contribute by improving the model or documentation!
