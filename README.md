# NASA Turbofan Engine Remaining Useful Life (RUL) Prediction

## Project Overview

This project implements a **predictive maintenance system** to estimate the **Remaining Useful Life (RUL)** of aircraft turbofan engines using **multivariate time-series sensor data**.

Using the **NASA CMAPSS Turbofan Engine Dataset**, several machine learning models were trained and compared before implementing a **Long Short-Term Memory (LSTM)** deep learning model to capture temporal degradation patterns in engine sensor data.

The objective is to predict how many operational cycles remain before an engine fails, enabling **proactive maintenance and failure prevention**.

---

## Dataset

The dataset used is the **NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS)**.

Each engine contains multiple cycles of sensor measurements until failure.

### Dataset Structure

Each row contains:

* Engine ID
* Time cycle
* 3 operational settings
* 21 sensor measurements

As engines operate over time, the sensor readings change due to **mechanical wear and degradation**.

The goal is to use these sensor readings to predict the **Remaining Useful Life (RUL)** of the engine.

---

## Project Workflow

### 1. Data Loading

The turbofan dataset was loaded and column names were assigned for:

* engine id
* cycle number
* operational settings
* sensor measurements

---

### 2. RUL Label Generation

For the training dataset, the Remaining Useful Life was computed using:

RUL = Maximum Cycle of Engine − Current Cycle

This creates the target variable that the model learns to predict.

---

### 3. Data Cleaning and Sensor Selection

Some sensors in the dataset contain:

* constant values
* near constant values
* no degradation patterns

Such sensors were removed to reduce noise and improve model performance.

---

### 4. Feature Scaling

Sensor measurements have different numerical ranges. Feature scaling was applied to normalize the values and ensure stable training of machine learning and deep learning models.

---

### 5. Sequence Generation

Since engine degradation is **time dependent**, the dataset was converted into sequences using a **sliding window approach**.

Example:

Cycles 1–30 → Predict RUL
Cycles 2–31 → Predict RUL
Cycles 3–32 → Predict RUL

This converts the dataset into a **3D structure required for LSTM models**:

(samples, time_steps, features)

---

## Models Trained and Compared

Before implementing the LSTM model, several baseline machine learning models were trained to evaluate performance and establish a benchmark.

### Models Used

* Random Forest Regressor
* Gradient Boosting Regressor
* Recurrent Neural Network (RNN)
* LSTM Neural Network

These models were trained using the processed sensor features and their performance was compared using regression metrics.

This comparison helped demonstrate that **deep learning models designed for sequential data perform better at capturing engine degradation patterns**.

---

## LSTM Model Architecture

The final model uses a Long Short-Term Memory network to capture temporal dependencies in sensor data.

Architecture:

LSTM Layer
Dropout Layer
Dense Output Layer

The LSTM learns how sensor measurements evolve over time before engine failure.

---

## Model Training

Training configuration:

Optimizer: Adam
Loss Function: Mean Squared Error (MSE)

The model minimizes the difference between predicted RUL and actual RUL using backpropagation.

---

## Prediction

For each engine in the test dataset:

* The last sequence of cycles is extracted
* The trained model predicts the Remaining Useful Life

This simulates real-world predictive maintenance scenarios where engines have not yet failed.

---

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## Applications

This project demonstrates **predictive maintenance**, which is widely used in:

* Aviation
* Manufacturing
* Industrial machinery
* Energy systems

Predicting Remaining Useful Life helps organizations:

* reduce downtime
* prevent unexpected failures
* optimize maintenance schedules
* reduce operational costs

---

## Future Improvements

Possible improvements include:

* GRU-based sequence models
* CNN + LSTM hybrid architectures
* Attention mechanisms
* Transformer-based models
* Hyperparameter tuning

