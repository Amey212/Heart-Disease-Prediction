# Heart Disease Prediction

## Overview
This project is a web-based application that predicts the likelihood of heart disease based on user input. It uses machine learning techniques and is implemented using Python, Flask, and a trained model.

## Features
- User-friendly web interface
- Takes user health data as input
- Uses a machine learning model for prediction
- Displays results dynamically

## Technologies Used
- Python
- Flask
- Scikit-Learn
- Multilayer Perceptron
- HTML, CSS, JavaScript
- Node.js (for front-end dependencies)

## Machine Learning Model
The heart disease prediction model is built using the Multilayer Perceptron algorithm. The dataset used for training is preprocessed using feature scaling, and the model is trained with the following configuration:

- **Algorithm**: Multilayer Pereceptron
- **hidden_layer_sizes**: (100, 50)
- **activation**: 15
- **solver**: 4
- **batch_size**='auto'
- **learning_rate**='adaptive',     
- **max_iter**=500,                 
- **random_state**=42,
- **early_stopping**=True,         
- **validation_fraction**=0.1 

The dataset is split into training and testing sets (80-20 split), and the model achieves high accuracy in predicting heart disease.

The trained model and scaler are saved using `joblib` for later use in making predictions.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Amey212/Heart-Disease-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Heart-Disease-Prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Open a browser and visit:
   ```
   http://127.0.0.1:5000/
   ```
3. Enter the required details and get predictions.

## Folder Structure
- `app.py` - Main application file
- `static/css` - Contains CSS files for styling
- `templates/` - HTML templates for web pages
- `model.pkl` - Trained machine learning model
- `rf_model.joblib` - Saved Random Forest model
- `scaler.joblib` - Saved scaler for feature normalization
- `requirements.txt` - List of required dependencies

## Contributing
Feel free to fork this repository and submit pull requests for improvements.



