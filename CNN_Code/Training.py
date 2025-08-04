import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv1D, GlobalAveragePooling1D # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Clean the data
    data_clean = data.dropna()

    # Select specific features for X and target variables for y
    features = ['a_x', 'a_y', 'a_z', 'a_sp', 'v_x', 'v_y', 'v_z', 'v_sp', 'pos_x', 'pos_y', 'pos_z', 'pos_sp']
    X = data_clean[features]
    y = data_clean[['curr_x', 'curr_y', 'curr_z']]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=8, activation='relu', padding='same'),
        Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        GlobalAveragePooling1D(),
        Dense(100, activation='relu'),
        Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    return history
def plot_loss(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.grid(True)
  plt.show()


def main():
    # Load and preprocess the data
    X_scaled, y_clean = load_and_preprocess_data(r'C:\Users\USER\Desktop\exercises\AMS LAB\TransferLearningMethodsOnTimeSeries\dataset\Data\DMG_CMX_600V\AL2007_Bauteil_1\CMX_Alu_Tr_Mat_1_alldata_allforces_MRR_allcurrent.csv')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, random_state=42)
    print("Train samples:", X_train.shape)
    print("Test samples:", X_test.shape)

    # Reshape X_train and X_test to include the sequence dimension
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Build and train the model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    print('Training the model...')
    history = train_model(model, X_train, y_train, X_test, y_test)
    print('Model trained successfully')

    # Optionally, save the trained model
    model.save('trained_model.h5')
    print("Plot")
    plot_loss(history)

if __name__ == "__main__":
    main()
