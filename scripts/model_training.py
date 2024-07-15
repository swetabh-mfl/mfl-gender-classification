from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from scripts.data_preparation import prepare_data
from scripts.model_building import build_model

def train_model(train_dir, validation_dir, model_save_path, epochs=10, batch_size=32):
    train_generator, validation_generator = prepare_data(train_dir, validation_dir)
    model = build_model()

    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )

    return history

if __name__ == "__main__":
    train_dir = 'data/train'
    validation_dir = 'data/validation'
    model_save_path = 'model/gender_classification_model.h5'
    epochs = 10
    batch_size = 32
    train_model(train_dir, validation_dir, model_save_path, epochs, batch_size)

