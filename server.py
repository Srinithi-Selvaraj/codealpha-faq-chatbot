from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faqs import faqs

app = Flask(__name__)
CORS(app)  # Enable CORS

# Convert all FAQ questions to a list
questions = [faq["question"] for faq in faqs]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_best_answer(user_question):
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)[0]
    best_index = similarities.argmax()
    return faqs[best_index]["answer"]

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("question")
    response = get_best_answer(user_message)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
