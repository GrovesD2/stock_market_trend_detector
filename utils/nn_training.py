import numpy as np

# NN imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Global variables
INPUT_LOC = 'data/nn_inputs/'
MODEL_LOC = 'nn_models/'

def get_model(x_train: np.array):
    
    return Sequential([
        Bidirectional(
            LSTM(128, input_shape = (x_train.shape[1], x_train.shape[2]),
                 return_sequences = True),
        ),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences = True)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(64),
        Dense(3, activation = 'softmax'),
    ])

def get_data():
    return (
        np.load(INPUT_LOC + 'x_train.npy'),
        np.load(INPUT_LOC + 'y_train.npy'),
        np.load(INPUT_LOC + 'x_test.npy'),
        np.load(INPUT_LOC + 'y_test.npy'),
    )

def model_callback_save(model_name: str):
    return ModelCheckpoint(
        filepath = MODEL_LOC + model_name,
        monitor = 'val_accuracy',
        mode = 'max',
        save_best_only=True,
    )

def evaluate_training(model,
                      x_test: np.array,
                      y_test: np.array):
    
    score = model.evaluate(
        x_test,
        y_test,
        verbose = 0,
    )

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    pred = np.argmax(
        model.predict(x_test),
        axis = 1,
    )
    
    cm = confusion_matrix(
        y_true = y_test,
        y_pred = pred
    )
    cm_scaled = cm/cm.astype(np.float).sum(axis = 0)
    
    disp = ConfusionMatrixDisplay(confusion_matrix = cm_scaled)
    disp.plot()
    
    return
