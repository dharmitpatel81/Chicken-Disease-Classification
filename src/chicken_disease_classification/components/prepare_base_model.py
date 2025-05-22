import tensorflow as tf
from chicken_disease_classification.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        """ 
        Loads a pre-trained VGG16 model without the top (classification) layer,
        as specified by config, and saves it.
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Modifies the base model to add a custom classification head and compiles it.
        
        Args:
            model: The base pre-trained model
            classes: Number of output classes
            freeze_all: If True, freeze all layers of the base model
            freeze_till: Freeze all layers except the last `freeze_till` layers
            learning_rate: Learning rate for training

        Returns:
            full_model: A compiled Keras model ready for training
        """

        # Option to freeze all layers (transfer learning without fine-tuning)
        if freeze_all:
            for layer in model.layers:
                model.trainable = False

        # Option to freeze all but the last `freeze_till` layers (fine-tuning)
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
                
        # Add a Flatten layer to convert final CNN features into a vector
        flatten_in = tf.keras.layers.Flatten()(model.output)

        # Add a Dense (fully connected) output layer with sigmoid activation for binary classification
        prediction = tf.keras.layers.Dense(
            units=1,
            activation="sigmoid"
        )(flatten_in)

        # Create the new model connecting base model input to new prediction layer
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model with SGD optimizer and categorical crossentropy loss
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"]
        )

        # Print a summary of the model architecture
        full_model.summary()
        return full_model

    def update_base_model(self):
        """
        Updates the base model by adding a custom classification head
        and saves the new model for training.
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        # Save the full model with new classification head
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)