from flask import Flask, request, jsonify
from model.predict_naive_bayes import classify as classify_bayes
from model.predict_bert import classify as classify_bert

classificationFunctions = {
    'bert': classify_bert,
    'naive_bayes': classify_bayes
}

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def http_classify():
    data = request.get_json()
    model = data.get('model', None)
    if not model:
        return jsonify({'error': 'Missing model type'}), 400
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Missing text'}), 400
    func = classificationFunctions.get(model)
    label, confidence = func(text)
    result = {'label': label, 'confidence': confidence}
    return jsonify(result), 200

if __name__ == '__main__':
    app.run()
