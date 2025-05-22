import tensorflow as tf
from chicken_disease_classification.entity.config_entity import TrainingConfig
from pathlib import Path
tf.config.run_functions_eagerly(True)

class Training:
    def __init__(self, config: TrainingConfig):
        # Save the configuration object which holds all training parameters
        self.config = config

    def get_base_model(self):
        # Load the base model that was saved after adding the classification head
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path, compile=False)

        # Recompile to reset optimizer for current model variables
        self.model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"]
        )

    def train_valid_generator(self):
        # Common preprocessing: Normalize pixel values to [0, 1] and Split 20% of data for validation
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,  
            validation_split=0.20,   
        )

        # Image size and batch settings for the generator
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Resize image to (height, width)
            batch_size=self.config.params_batch_size,        # Number of images per batch
            interpolation="bilinear"                         # Interpolation used for resizing
        )

        # Validation generator without augmentation
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,   # Folder structure: class subdirs inside this
            subset="validation",                   # Use 20% for validation
            shuffle=False,                         # Keep order stable for evaluation
            class_mode="binary",                   # Classify images as either 0 or 1
            **dataflow_kwargs
        )

        # Training generator with optional data augmentation
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,              # Random rotation
                horizontal_flip=True,           # Flip images horizontally
                width_shift_range=0.2,          # Horizontal shift
                height_shift_range=0.2,         # Vertical shift
                shear_range=0.2,                # Shear transformation
                zoom_range=0.2,                 # Random zoom
                **datagenerator_kwargs
            )
        else:
            # Use same generator as validation if no augmentation is enabled
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,   # Same data directory
            subset="training",                     # Use 80% for training
            shuffle=True,                          # Shuffle for better learning
             class_mode="binary",                   # Classify images as either 0 or 1
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Save the complete model (architecture + weights)
        model.save(path)

    def train(self, callback_list: list):
        
        # Calculate the number of batches per epoch for both training and validation
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Train the model using the generators and provided callbacks
        self.model.fit(
            self.train_generator,                   # Training generator
            epochs=self.config.params_epochs,       # Total number of epochs to train
            steps_per_epoch=self.steps_per_epoch,   # How many batches per epoch
            validation_steps=self.validation_steps, # Validation steps per epoch
            validation_data=self.valid_generator,   # Validation generator
            callbacks=callback_list                 # List of Keras callbacks (e.g., early stopping)
        )

        # Save the fully trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )