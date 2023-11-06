from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Carga el modelo de inteligencia artificial fuera de las rutas
model = keras.models.load_model("digit_classification.h5")

# Define la ruta para enviar imágenes al programa de inteligencia artificial
@app.route("/predict", methods=["POST"])
def predict():
    # Obtiene la imagen del usuario
    image = request.files["image"]

    # Preprocesa la imagen
    image = image.read()
    image = tf.image.decode_png(image)
    image = tf.image.rgb_to_grayscale(image)
    image = tf.image.resize(image, (28, 28))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32) / 255.0

    # Realiza la predicción
    prediction = model.predict(image)

    # Obtén el índice de la clase con la probabilidad más alta
    predicted_class = tf.argmax(prediction[0]).numpy()

    # Devuelve la predicción al usuario en formato JSON
    return jsonify({"prediction": int(predicted_class)})

# Define la ruta para la página principal
@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    # Inicia la aplicación Flask
    app.run(debug=True)
