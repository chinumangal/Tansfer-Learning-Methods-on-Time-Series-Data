import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data_clean = data.dropna()
    X_clean = data_clean.drop(['curr_x', 'curr_y', 'curr_z'], axis=1)
    y_clean = data_clean[['curr_x', 'curr_y', 'curr_z']]
    return X_clean, y_clean

def scale_data(scaler, X_clean, feature_names):
    # Ensure only the expected 12 features are selected
    expected_features = feature_names[:12]  
    for col in expected_features:
        if col not in X_clean.columns:
            X_clean[col] = 0
    X_clean = X_clean[expected_features]
    X_scaled = scaler.fit_transform(X_clean)
    return X_scaled

def reshape_data(X_scaled):
    print(f"Original shape: {X_scaled.shape}")
    num_samples = X_scaled.shape[0]
    num_features = X_scaled.shape[1]
    if num_features != 12:
        raise ValueError(f"Expected 12 features, but got {num_features}")
    X_scaled = X_scaled.reshape((num_samples, 12, 1))
    return X_scaled

def evaluate_model(model, X_scaled, y_clean):
    loss = model.evaluate(X_scaled, y_clean, verbose=0)
    y_pred = model.predict(X_scaled)
    return loss, y_pred

def calculate_rmse(y_clean, y_pred):
    rmse = np.sqrt(mean_squared_error(y_clean, y_pred, multioutput='raw_values'))
    return rmse

def calculate_percentage_deviation(y_clean, y_pred):
    percent_dev = 100 * (y_pred - y_clean.values) / y_clean.values
    percent_dev_df = pd.DataFrame(percent_dev, columns=['Percent_Dev_curr_x', 'Percent_Dev_curr_y', 'Percent_Dev_curr_z'])
    return percent_dev_df

def plot_actual_vs_predicted(y_clean, y_pred, rmse):
    plt.figure(figsize=(12, 8))
    plt.scatter(y_clean['curr_x'], y_pred[:, 0], color='blue', alpha=0.5, label=f'curr_x (RMSE: {rmse[0]:.2f})')
    plt.scatter(y_clean['curr_y'], y_pred[:, 1], color='green', alpha=0.5, label=f'curr_y (RMSE: {rmse[1]:.2f})')
    plt.scatter(y_clean['curr_z'], y_pred[:, 2], color='orange', alpha=0.5, label=f'curr_z (RMSE: {rmse[2]:.2f})')
    min_val = min(y_clean.min().min(), y_pred.min())
    max_val = max(y_clean.max().max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, linestyle='--', label='Ideal Fit')
    plt.title('Actual vs. Predicted Values for curr_x, curr_y, curr_z', fontsize=16)
    plt.xlabel('Actual', fontsize=16)
    plt.ylabel('Predicted', fontsize=16)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

def plot_percentage_deviation_distribution(percent_dev_df, rmse):
    plt.figure(figsize=(12, 6))
    plt.hist(percent_dev_df['Percent_Dev_curr_x'], bins=50, alpha=0.5, label='curr_x')
    plt.hist(percent_dev_df['Percent_Dev_curr_y'], bins=50, alpha=0.5, label='curr_y')
    plt.hist(percent_dev_df['Percent_Dev_curr_z'], bins=50, alpha=0.5, label='curr_z')
    plt.title('Distribution of Percentage Deviations', fontsize=15)
    plt.xlabel('Percentage Deviation', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.legend(loc='upper left')
    plt.axvline(x=percent_dev_df['Percent_Dev_curr_x'].mean(), color='blue', linestyle='--', label=f'Mean Percent Dev curr_x (RMSE: {rmse[0]:.2f})')
    plt.axvline(x=percent_dev_df['Percent_Dev_curr_y'].mean(), color='green', linestyle='--', label=f'Mean Percent Dev curr_y (RMSE: {rmse[1]:.2f})')
    plt.axvline(x=percent_dev_df['Percent_Dev_curr_z'].mean(), color='orange', linestyle='--', label=f'Mean Percent Dev curr_z (RMSE: {rmse[2]:.2f})')
    plt.legend(loc='upper right')
    plt.show()

def plot_normal_percentage_deviation_distribution(percent_dev_new_df):
    """
    Plot distribution of percentage deviations for each target variable using a line plot.

    Args:
        percent_dev_new_df (pd.DataFrame): Percentage deviations.
    """
    plt.figure(figsize=(12, 6))

    # Plotting percentage deviations for curr_x
    plt.plot(percent_dev_new_df.index, percent_dev_new_df['Percent_Dev_curr_x'], label='curr_x', color='blue', marker='o')

    # Plotting percentage deviations for curr_y
    plt.plot(percent_dev_new_df.index, percent_dev_new_df['Percent_Dev_curr_y'], label='curr_y', color='green', marker='o')

    # Plotting percentage deviations for curr_z
    plt.plot(percent_dev_new_df.index, percent_dev_new_df['Percent_Dev_curr_z'], label='curr_z', color='orange', marker='o')

    plt.title('Percentage Deviation for Each Target Variable', fontsize=15)
    plt.xlabel('Time steps', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def plot_actual_vs_predicted_over_time(y_clean, y_pred):
    """
    Plot actual vs predicted values over time steps.

    Args:
        y_clean (pd.DataFrame): Actual values.
        y_pred (np.ndarray): Predicted values.
    """
    plt.figure(figsize=(12, 8))

    # Plotting actual vs predicted for curr_x
    plt.plot(y_clean.index, y_clean['curr_x'], label='Actual curr_x', color='blue')
    plt.plot(y_clean.index, y_pred[:, 0], label='Predicted curr_x', color='blue', linestyle='--')

    # Plotting actual vs predicted for curr_y
    plt.plot(y_clean.index, y_clean['curr_y'], label='Actual curr_y', color='green')
    plt.plot(y_clean.index, y_pred[:, 1], label='Predicted curr_y', color='green', linestyle='--')

    # Plotting actual vs predicted for curr_z
    plt.plot(y_clean.index, y_clean['curr_z'], label='Actual curr_z', color='orange')
    plt.plot(y_clean.index, y_pred[:, 2], label='Predicted curr_z', color='orange', linestyle='--')

    plt.title('Actual vs. Predicted Values Over Time Steps', fontsize=16)
    plt.xlabel('Time Steps', fontsize=16)
    plt.ylabel('Values', fontsize=16)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def calculate_rmse_and_percentage_deviation_for_each_sample(y_clean, y_pred):
    rmse_x_list, rmse_y_list, rmse_z_list = [], [], []
    percent_x_list, percent_y_list, percent_z_list = [], [], []

    for i in range(len(y_clean)):
        actual_x, actual_y, actual_z = y_clean.iloc[i]
        pred_x, pred_y, pred_z = y_pred[i]

        rmse_x_i = np.sqrt(mean_squared_error([actual_x], [pred_x]))
        rmse_y_i = np.sqrt(mean_squared_error([actual_y], [pred_y]))
        rmse_z_i = np.sqrt(mean_squared_error([actual_z], [pred_z]))

        percent_dev_x_i = round((abs(pred_x - actual_x) / actual_x) * 100, 2) if actual_x != 0 else 0
        percent_dev_y_i = round((abs(pred_y - actual_y) / actual_y) * 100, 2) if actual_y != 0 else 0
        percent_dev_z_i = round((abs(pred_z - actual_z) / actual_z) * 100, 2) if actual_z != 0 else 0

        rmse_x_list.append(rmse_x_i)
        rmse_y_list.append(rmse_y_i)
        rmse_z_list.append(rmse_z_i)

        percent_x_list.append(percent_dev_x_i)
        percent_y_list.append(percent_dev_y_i)
        percent_z_list.append(percent_dev_z_i)

    return pd.DataFrame({
        'RMSE_curr_x': rmse_x_list,
        'RMSE_curr_y': rmse_y_list,
        'RMSE_curr_z': rmse_z_list,
        'Percent_Dev_curr_x': percent_x_list,
        'Percent_Dev_curr_y': percent_y_list,
        'Percent_Dev_curr_z': percent_z_list
    })

def main():
    X_clean, y_clean = load_and_preprocess_data(r'C:\Users\USER\Desktop\exercises\AMS LAB\TransferLearningMethodsOnTimeSeries\dataset\Data\DMG_CMX_600V\AL2007_Bauteil_1_Aircut\CMX_Alu_Tr_Air_1_alldata_allcurrent.csv')
    scaler = StandardScaler()
    scaler.fit(X_clean)
    feature_names = X_clean.columns.tolist()
    X_scaled = scale_data(scaler, X_clean, feature_names)
    X_scaled = reshape_data(X_scaled)
    model = tf.keras.models.load_model('trained_model.h5')
    loss, y_pred = evaluate_model(model, X_scaled, y_clean)
    print(f'Test Loss on new data: {loss}')
    rmse = calculate_rmse(y_clean, y_pred)
    print(f'Test RMSE on new data: {rmse}')
    percent_dev_df = calculate_percentage_deviation(y_clean, y_pred)
    print("Percentage Deviations (first 5 samples):")
    print(percent_dev_df.head())
    plot_actual_vs_predicted(y_clean, y_pred, rmse)
    plot_percentage_deviation_distribution(percent_dev_df, rmse)
    plot_normal_percentage_deviation_distribution(percent_dev_df)
    plot_actual_vs_predicted_over_time(y_clean, y_pred)
    results_df = calculate_rmse_and_percentage_deviation_for_each_sample(y_clean, y_pred)
    print("RMSE and Percentage Deviation for each sample (first 5 samples):")
    print(results_df.head())
    results_df.to_csv('Aluminium2007aircut.csv', index=False)

if __name__ == "__main__":
    main()