from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from werkzeug.utils import secure_filename
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import base64
import numpy as np
import io

app = Flask(__name__)

base_directory = os.path.join(os.getcwd(), 'static', 'Data')

model_directory = os.path.join(os.getcwd(), 'static', 'model')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model = None

def train_model():
    num_classes = 7
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory('static/Data', target_size=(224, 224), batch_size=32, class_mode='categorical')

    base_model = MobileNetV2(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    global model
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=10)

    # Save the trained model
    model.save(os.path.join(model_directory, 'trained_model.h5'))

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/predict_button', methods=['POST'])
def predict_button():
    global model
    if request.method == 'POST':
        file = request.files['file']
        encoded_image = base64.b64encode(file.read()).decode('utf-8')

        img = Image.open(io.BytesIO(base64.b64decode(encoded_image))).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Ensure model is loaded before making predictions
        if model is None:
            model = load_model(os.path.join(model_directory, 'trained_model.h5'))

        prediction = model.predict(img)

        # Update class names based on subfolder names in static/Data
        subfolders = [folder for folder in os.listdir(base_directory) if
                      os.path.isdir(os.path.join(base_directory, folder))]
        class_names = sorted(subfolders)

        if len(class_names) > 0 and prediction.size > 0:
            max_index = np.argmax(prediction)
            if max_index < len(class_names):
                result = class_names[max_index]
                accuracy = float(prediction[0][max_index])
                return render_template('predict.html', prediction_result=result, prediction_accuracy=accuracy,
                                       encoded_image=encoded_image)
            else:
                return jsonify({'error': 'Invalid prediction index', 'prediction': prediction.tolist()})
        else:
            return jsonify({'error': 'Invalid prediction or class names', 'prediction': prediction.tolist()})

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        new_subfolder = request.form['new_subfolder']
        new_subfolder_path = os.path.join(base_directory, new_subfolder)
        if not os.path.exists(new_subfolder_path):
            os.makedirs(new_subfolder_path)

        for upload_file in request.files.getlist('file'):
            if upload_file.filename != '':
                filename = secure_filename(upload_file.filename)
                upload_file.save(os.path.join(new_subfolder_path, filename))

    subfolders = [folder for folder in os.listdir(base_directory) if
                  os.path.isdir(os.path.join(base_directory, folder))]

    subfolder_data = {}

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_directory, subfolder)

        images = [f'/static/Data/{subfolder}/{image}' for image in os.listdir(subfolder_path) if
                  image.lower().endswith(('.png', '.jpg', '.jpeg'))]

        subfolder_data[subfolder] = images

    return render_template('dataset.html', subfolder_data=subfolder_data)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    subfolder = request.form['subfolder']
    subfolder_path = os.path.join(base_directory, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    for upload_file in request.files.getlist('file'):
        if upload_file.filename != '':
            filename = secure_filename(upload_file.filename)
            upload_file.save(os.path.join(subfolder_path, filename))

    return jsonify({'message': 'Images uploaded successfully.'}), 200


@app.route('/delete_subfolder', methods=['POST'])
def delete_subfolder():
    data = request.get_json()
    if 'subfolder' not in data:
        return jsonify({'error': 'Subfolder key is missing in the request.'}), 400

    subfolder = data['subfolder']
    subfolder_path = os.path.join(base_directory, subfolder)

    try:
        if os.path.exists(subfolder_path):
            shutil.rmtree(subfolder_path)
            return jsonify({'message': 'Subfolder deleted successfully.'}), 200
        else:
            return jsonify({'error': 'Subfolder does not exist.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/showAll', methods=['GET', 'POST'])
def showAll():
    base_directory = os.path.join(os.getcwd(), 'static', 'Data')

    if not os.path.exists(base_directory):
        return "Error: The specified base directory does not exist."

    subfolders = [folder for folder in os.listdir(base_directory) if
                  os.path.isdir(os.path.join(base_directory, folder))]

    subfolder_data = {}

    for subfolder in subfolders:
        subfolder_path = os.path.join(base_directory, subfolder)

        images = [f'/static/Data/{subfolder}/{image}' for image in os.listdir(subfolder_path) if
                  image.lower().endswith(('.png', '.jpg', '.jpeg'))]

        subfolder_data[subfolder] = images

    return render_template('showAll.html', subfolder_data=subfolder_data)

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/train_button', methods=['POST'])
def train_button():
    train_model()
    return 'Модель сәтті оқытылды.'

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run()