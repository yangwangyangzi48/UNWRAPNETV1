from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from datetime import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth is enabled.")
    except RuntimeError as e:
        print(e)

def train_model(model, tr_dataset, val_dataset, epochs=30, batch_size=64, checkpoint_path=None):
    if checkpoint_path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = f'best_model_{timestamp}.h5'

    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f'Loaded weights from {checkpoint_path}')

    log_dir = "/root/tf-logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1)
    ]

    tr_dataset = tr_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    history = model.fit(tr_dataset,
                        epochs=epochs,
                        validation_data=val_dataset,
                        callbacks=callbacks,
                        verbose=1,
                        batch_size=batch_size)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # if 'accuracy' in history.history:
    #     plt.subplot(1, 2, 2)
    #     plt.plot(history.history['accuracy'], label='Training Accuracy')
    #     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    #     plt.title('Accuracy Curve')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.legend()

    plt.show()

    return history