from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load GoEmotions model
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=3)

@app.route("/", methods=["GET", "POST"])
def index():
    emotions = []
    text = ""

    if request.method == "POST":
        text = request.form["sentence"]
        if text.strip() == "":
            emotions = [{"label": "Error", "score": "Please enter a valid sentence."}]
        else:
            result = emotion_classifier(text)[0]  # top_k=3 → returns a list
            emotions = [{"label": r["label"], "score": f"{r['score']*100:.2f}%"} for r in result]

    return render_template("index.html", emotions=emotions, text=text)

if __name__ == "__main__":
    app.run(debug=True)
