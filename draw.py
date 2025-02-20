from flask import Flask, request, jsonify, render_template
from DECIMER import predict_SMILES
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('draw.html')

@app.route('/get-smile', methods=['POST'])
def get_smile():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image_path = os.path.join("uploads", image_file.filename)
    
    # Ensure the uploads directory exists
    os.makedirs("uploads", exist_ok=True)
    
    image_file.save(image_path)
    
    try:
        smiles = predict_SMILES(image_path)
        return jsonify({"smiles": smiles})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)