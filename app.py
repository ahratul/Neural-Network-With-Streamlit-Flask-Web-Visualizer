import streamlit as st
import requests
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np
from flask import Flask, request

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')



    app = Flask(__name__)

    model = tf.keras.models.load_model('model.h5')
    feature_model = tf.keras.models.Model(
        model.inputs,
        [layer.output for layer in model.layers]
    )

    _, (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_test = x_test / 255


    def get_prediction():
        index = np.random.choice(x_test.shape[0])
        image = x_test[index, :, :]
        image_arr = np.reshape(image, (1, 784))
        return feature_model.predict(image_arr), image


    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            preds, image = get_prediction()
            final_preds = [p.tolist() for p in preds]
            return json.dumps({
                'prediction': final_preds,
                'image': image.tolist()
            })

        return 'Welcome To The Model Server!'


    if __name__ == '__main__':
        app.run()

    # @app.route('/foo')
    # def serve_foo():
    #     return 'This page is served via Flask!'
    #
    # app.run(port=8888)


# We'll never reach this part of the code the first time this file executes!


URI = 'http://127.0.0.1:5000'
st.title('Neural Network Visualizer WebApp')

if st.button('Get Prediction !'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))

    st.image(image, width=150)

    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32, 4))

        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16

        for i, number in enumerate(numbers):
            plt.subplot(row, col, i + 1)
            plt.imshow(number * np.ones((8, 8, 3)).astype('float64'))
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize=40)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text('Layer {}'.format(layer + 1))
        st.pyplot()
