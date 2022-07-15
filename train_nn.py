from utils import nn_preprocess, nn_training
from tensorflow.keras.optimizers import Adam


CONFIG = {
    # Generate data config settings
    'generate input data': True,
    'days back': 30,
    'train split': 0.8,
    
    # NN training configs
    'train nn': True,
    'epochs': 10,
    'batch': 8,
    'validation': 0.1,
    'learn rate': 1e-3,
    'model name': 'trend_detector'    
}

if __name__ == '__main__':

    if CONFIG['generate input data']:
        nn_preprocess.main(CONFIG)
    
    if CONFIG['train nn']:
    
        # Get training/testing data
        x_train, y_train, x_test, y_test = nn_training.get_data()
        
        # Get the nn model architecture
        model = nn_training.get_model(x_train)
        
        # Compile the model
        model.compile(
            loss = 'sparse_categorical_crossentropy',
            optimizer = Adam(learning_rate = CONFIG['learn rate']),
            metrics = ['accuracy']
        )
        
        # Fit the model
        model.fit(
            x_train,
            y_train,
            epochs = CONFIG['epochs'],
            batch_size = CONFIG['batch'],
            validation_split = CONFIG['validation'],
            callbacks = [nn_training.model_callback_save(CONFIG['model name'])]
        )
        
        # Evaluate the training
        nn_training.evaluate_training(model, x_test, y_test)
