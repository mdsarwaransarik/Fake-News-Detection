from flask import Flask, render_template, request
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load tokenizer and model
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

model = load_model('fake_news_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy')

# Maximum sequence length (adjust this according to your training parameters)
max_sequence_length = 500

# Function to preprocess input text
def preprocess_input_text(text, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return padded_sequences

# Define routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the input text from the form
        input_text = request.form["news_text"]

        # Preprocess and make prediction
        processed_text = preprocess_input_text(input_text, tokenizer, max_sequence_length)
        prediction = (model.predict(processed_text) > 0.5).astype("int32")

        # Output result based on prediction
        if prediction[0][0] == 1:
            result = "Fake News"
        else:
            result = "Real News"

        return render_template("index.html", prediction=result)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
