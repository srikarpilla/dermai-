
DermAI - AI-Based Skin Condition Analyzer

DermAI is a web application designed to assist in the preliminary identification of common skin conditions using deep learning. Users can upload an image of the affected skin area and optionally select accompanying symptoms. The system combines image analysis with symptom matching to provide a more informed prediction.

This project was developed as part of a final-year computer science curriculum to explore the application of convolutional neural networks in medical image classification.

Features

- Image-based prediction using a fine-tuned EfficientNetB0 model trained on the DermNet dataset (23 classes).
- Symptom input via checkboxes for improved contextual analysis.
- Symptom matching against a predefined database to show alignment between user-reported symptoms and typical disease profiles.
- Confidence scoring from the model.
- Responsive, user-friendly interface with sections for analysis, model details, and condition explanations.
- Local processing with no data storage for privacy.

Requirements

- Python 3.10 or higher
- TensorFlow (CPU version)
- Flask
- Pillow
- NumPy

A completerequirements.txt is included in the repository.

Installation

1. Clone the repository:
  
   git clone https://github.com/yourusername/dermai-skin-disease-app.git
   cd dermai-skin-disease-app
  

2. (Recommended) Create and activate a virtual environment:
  
   python -m venv venv
   venv\Scripts\activate    Windows
   source venv/bin/activate macOS/Linux
  

3. Install dependencies:
  
   pip install -r requirements.txt
  

4. Ensure the following files are in the project root:
   -best_weights.weights.h5
   -model_architecture.json
   -class_names.json
   -symptoms.json

Usage

1. Start the server:
  
   python backend.py
  

2. Open a web browser and navigate tohttp://127.0.0.1:5000.

3. On the main page:
   - Select any relevant symptoms (optional).
   - Upload a clear, well-lit photo of the skin area.
   - View the prediction, confidence level, and symptom match analysis.

Model Details

- Architecture: EfficientNetB0 base with a custom classification head.
- Dataset: DermNet (approximately 19,000 images across 23 skin condition classes).
- Training: Transfer learning with data augmentation; class weights applied to handle imbalance.
- Performance: Approximately 41% top-1 accuracy and 75% top-5 accuracy on held-out test data.
- Input: 224×224 RGB images, normalized to [-1, 1] range.

Symptom data is sourced from publicly available medical references and is used only for educational comparison.

Important Disclaimer

DermAI is an academic project intended solely for educational and research purposes. It is not a medical device and must not be used for diagnosis or treatment decisions.

Predictions may be incorrect. Skin conditions often require professional evaluation, including clinical history and, if necessary, biopsy or other tests.

Always consult a qualified dermatologist or physician for any skin concerns.

Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
