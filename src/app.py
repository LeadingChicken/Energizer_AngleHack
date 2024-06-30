from flask import Flask, request, jsonify
import os
from metrics.formulas.calculate_metrics import calculate_metrics

app = Flask(__name__)


@app.route('/analyze', methods=['POST'])
def calculate_metrics_api():
    data = request.json
    folder_path = data.get('folder_path')

    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({"error": "Invalid folder path"}), 400

    img_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not img_list:
        return jsonify({"error": "No images found in the folder"}), 400

    metrics = calculate_metrics(img_list)
    return jsonify(metrics)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
