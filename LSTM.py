from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
import loadcsv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def create_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(18, 34)))
    model.add(LSTM(512, return_sequences=True))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(loss="binary_crossentropy" , optimizer= "adam", metrics=['accuracy'])
    return model

def train():
    model = create_model()
    model.summary()
    checkpoint = ModelCheckpoint('best_model_weights.h5', 
                                  monitor='val_accuracy', 
                                  save_best_only=True,      
                                  save_weights_only=True,  
                                  mode='max')
    
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   patience=20,
                                   verbose=0,
                                   restore_best_weights=True)
    
    history = model.fit(train_X, train_y,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.25,
                  callbacks=[checkpoint, early_stopping])
    return model, history

def load(weight='best_model_weights.h5'):
    model = create_model()
    model.load_weights(weight)
    return model

def plot():
    plt.figure(1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Vali'], loc='upper left')
    plt.show()
    
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Vali'], loc='upper left')
    plt.show()

    plt.figure(3)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=["Fall","NotFall"], yticklabels=["Fall","NotFall"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    dataset_folder = "CSV_Dataset"  
    label_n_path = loadcsv.get_dataset(dataset_folder)
    numeric_label_n_path = loadcsv.numericalize(label_n_path)
    train_X, train_y, test_X, test_y = loadcsv.random_split(numeric_label_n_path)
    epochs = 500
    batch_size = 50
    n_timesteps, n_features, n_outputs =  train_X.shape[1], train_X.shape[2], train_y.shape[1]
    
    model, history = train()
    y_pred = model.predict(test_X)
    y_pred_classes = []
    for i in y_pred:
        if i < 0.5:
            y_pred_classes.append(0)
        else:
            y_pred_classes.append(1)
    y_pred_classes = np.array(y_pred_classes)
    confusion_mtx = confusion_matrix(test_y, y_pred_classes)
    _, accuracy =  model.evaluate(test_X, test_y, batch_size=len(test_X), verbose=1) 
    print("-"*50,f"\nACCURACY:{accuracy}\n","-"*50)
    
    plot()


