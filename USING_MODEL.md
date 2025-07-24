# Using Your MNIST Model in a Real Application

## Saving the Model
After training your model, save it to disk using the H5 format:

```python
model.save("mnist_model.h5")
```

## What is H5?
H5 (or HDF5) is a file format for storing large, complex data. In Keras/TensorFlow, saving a model as `.h5` stores the model architecture, weights, and optimizer state in a single file. This makes it easy to reload and use the model later, or share it with others.

## Serving the Model as an API
To use your model in a real application, you can create a web service (API) that loads the model and predicts on new data. Here’s a simple example using Flask:

1. **Install Flask** (if not already):
   ```sh
   pip install flask
   ```

2. **Create `app.py`**:
   ```python
   from flask import Flask, request, jsonify
   import numpy as np
   import tensorflow as tf

   app = Flask(__name__)
   model = tf.keras.models.load_model("mnist_model.h5")

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.get_json(force=True)
       img = np.array(data['image']).reshape(1, 28, 28) / 255.0
       prediction = model.predict(img)
       predicted_class = int(np.argmax(prediction[0]))
       return jsonify({'prediction': predicted_class})

   if __name__ == '__main__':
       app.run(debug=True)
   ```

3. **Run the API**:
   ```sh
   python app.py
   ```

4. **Send a Prediction Request**:
   Send a POST request to `http://localhost:5000/predict` with a JSON body like:
   ```json
   {"image": [[...28 values...], [...], ..., [...]]}
   ```
   The `image` should be a 28x28 grayscale array (list of lists).

## Summary
- Save your model as `.h5` after training.
- Use Flask (or FastAPI) to serve predictions via an API.
- The `.h5` file makes it easy to reload and use your trained model in any Python application.
