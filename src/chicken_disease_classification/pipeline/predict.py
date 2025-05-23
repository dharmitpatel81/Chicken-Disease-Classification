import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts", "training", "model.keras"))

        imagename = self.filename
        test_image = load_img(imagename, target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # normalize if  model was trained with rescale

        result = model.predict(test_image)  # result is a probability between 0 and 1
        print(result)

        if result[0][0] >= 0.5:
            prediction = "Healthy"
        else:
            prediction = "Coccidiosis"
        return [{"image": prediction}]