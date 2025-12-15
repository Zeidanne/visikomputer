from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global KNN model storage
knn_model = None
knn_features = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_glcm_features(image_path, distances, angles, levels):
    """
    Extract GLCM texture features from an image
    
    Parameters:
    - image_path: path to the image file
    - distances: list of pixel pair distance offsets
    - angles: list of pixel pair angles in radians
    - levels: number of gray levels
    
    Returns:
    - Dictionary containing GLCM features
    """
    # Read and convert image to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalize to specified gray levels
    gray = (gray / 256 * levels).astype(np.uint8)
    
    # Compute GLCM
    glcm = graycomatrix(gray, distances=distances, angles=angles, 
                        levels=levels, symmetric=True, normed=True)
    
    # Extract features
    features = {
        'contrast': graycoprops(glcm, 'contrast').flatten().tolist(),
        'dissimilarity': graycoprops(glcm, 'dissimilarity').flatten().tolist(),
        'homogeneity': graycoprops(glcm, 'homogeneity').flatten().tolist(),
        'energy': graycoprops(glcm, 'energy').flatten().tolist(),
        'correlation': graycoprops(glcm, 'correlation').flatten().tolist(),
    }
    
    # Calculate average for each feature
    features_avg = {
        'contrast_avg': float(np.mean(features['contrast'])),
        'dissimilarity_avg': float(np.mean(features['dissimilarity'])),
        'homogeneity_avg': float(np.mean(features['homogeneity'])),
        'energy_avg': float(np.mean(features['energy'])),
        'correlation_avg': float(np.mean(features['correlation'])),
    }
    
    return {
        'features': features,
        'features_avg': features_avg,
        'image_shape': list(gray.shape)
    }

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Redirect to GLCM page"""
    return render_template('glcm.html')

@app.route('/glcm')
def glcm_page():
    """GLCM Feature Extraction page"""
    return render_template('glcm.html')

@app.route('/knn')
def knn_page():
    """KNN Classification page"""
    return render_template('knn.html')

# ============================================================
# GLCM API ENDPOINTS
# ============================================================

@app.route('/extract', methods=['POST'])
def extract():
    """Extract GLCM features from uploaded image"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, BMP, TIFF'}), 400
        
        # Get parameters
        distances_str = request.form.get('distances', '1')
        angles_str = request.form.get('angles', '0,45,90,135')
        levels = int(request.form.get('levels', '256'))
        
        # Parse distances and angles
        distances = [int(d.strip()) for d in distances_str.split(',')]
        angles_deg = [float(a.strip()) for a in angles_str.split(',')]
        angles = [np.radians(a) for a in angles_deg]
        
        # Validate levels
        if levels not in [8, 16, 32, 64, 128, 256]:
            return jsonify({'error': 'Gray levels must be one of: 8, 16, 32, 64, 128, 256'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features
        result = extract_glcm_features(filepath, distances, angles, levels)
        
        # Add parameters to result
        result['parameters'] = {
            'distances': distances,
            'angles': angles_deg,
            'levels': levels
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# KNN API ENDPOINTS
# ============================================================

@app.route('/knn/dataset-info', methods=['GET'])
def knn_dataset_info():
    """Get Iris dataset information and preview"""
    try:
        iris = load_iris()
        
        # Create preview data (first 10 samples)
        preview = []
        for i in range(10):
            preview.append({
                'sepal_length': float(iris.data[i][0]),
                'sepal_width': float(iris.data[i][1]),
                'petal_length': float(iris.data[i][2]),
                'petal_width': float(iris.data[i][3]),
                'target': int(iris.target[i])
            })
        
        # Feature statistics
        stats = {
            'n_samples': int(iris.data.shape[0]),
            'n_features': int(iris.data.shape[1]),
            'n_classes': len(iris.target_names),
            'feature_names': list(iris.feature_names),
            'target_names': list(iris.target_names),
            'preview': preview
        }
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knn/train', methods=['POST'])
def knn_train():
    """Train KNN model with specified parameters"""
    global knn_model, knn_features
    
    try:
        data = request.get_json()
        
        # Get parameters
        features_idx = data.get('features', [0, 1, 2, 3])
        k = data.get('k', 3)
        test_size = data.get('test_size', 0.2)
        metric = data.get('metric', 'euclidean')
        
        # Load and prepare data
        iris = load_iris()
        X = iris.data[:, features_idx]
        y = iris.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_model.fit(X_train, y_train)
        knn_features = features_idx
        
        # Predictions
        y_pred = knn_model.predict(X_test)
        
        # Metrics
        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average='weighted'))
        recall = float(recall_score(y_test, y_pred, average='weighted'))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Classification Report per class
        report = classification_report(y_test, y_pred, output_dict=True)
        class_report = []
        for i, class_name in enumerate(['0', '1', '2']):
            if class_name in report:
                class_report.append({
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1_score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                })
        
        # Visualization data (full dataset)
        vis_data = []
        for i in range(len(iris.data)):
            vis_data.append({
                'features': iris.data[i].tolist(),
                'target': int(iris.target[i])
            })
        
        return jsonify({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'visualization_data': vis_data,
            'model_params': {
                'k': k,
                'metric': metric,
                'features': features_idx,
                'test_size': test_size
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/knn/predict', methods=['POST'])
def knn_predict():
    """Predict class for new data point"""
    global knn_model, knn_features
    
    try:
        if knn_model is None:
            return jsonify({'error': 'Model belum di-train. Silakan train model terlebih dahulu.'}), 400
        
        data = request.get_json()
        features = data.get('features', [])
        
        if len(features) != 4:
            return jsonify({'error': 'Harus memasukkan 4 nilai fitur'}), 400
        
        # Select only the features used in training
        X = np.array([features])[: , knn_features]
        
        # Predict
        prediction = knn_model.predict(X)[0]
        
        # Get probabilities (for confidence)
        probabilities = knn_model.predict_proba(X)[0]
        confidence = float(probabilities[prediction])
        
        # Get nearest neighbors info
        distances, indices = knn_model.kneighbors(X)
        
        # Load iris to get neighbor classes
        iris = load_iris()
        neighbors = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            neighbors.append({
                'distance': float(dist),
                'class': int(iris.target[idx])
            })
        
        return jsonify({
            'predicted_class': int(prediction),
            'confidence': confidence,
            'probabilities': probabilities.tolist(),
            'neighbors': neighbors
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)