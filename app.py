
import pandas as pd
import numpy as np
from math import log2
from skimage.feature import graycomatrix, graycoprops
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2

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

# ============================================================
# DECISION TREE (PLAY GOLF) API ENDPOINTS
# ============================================================

# --- Helper Functions (Manual ID3) ---

def entropy(target_col):
    """Calculate entropy of a target column"""
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = 0
    for i in range(len(elements)):
        probability = counts[i] / np.sum(counts)
        entropy_value -= probability * log2(probability)
    return round(entropy_value, 3)

def info_gain(data, split_attribute_name, target_name="PlayGolf"):
    """Calculate information gain for a split"""
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    weighted_entropy = 0
    subsets_entropy = {}
    
    for i in range(len(vals)):
        subset = data[data[split_attribute_name] == vals[i]]
        if len(subset) == 0: continue
        subset_ent = entropy(subset[target_name])
        weighted_entropy += (counts[i] / np.sum(counts)) * subset_ent
        subsets_entropy[str(vals[i])] = subset_ent
    
    information_gain = total_entropy - weighted_entropy
    return round(information_gain, 3), total_entropy, subsets_entropy

def Id3(data, originaldata, features, target_attribute_name="PlayGolf", parent_node_class=None):
    """
    ID3 Algorithm to build Decision Tree manually
    Returns a dictionary representing the tree
    """
    # 1. If all target values are the same, return that value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return str(np.unique(data[target_attribute_name])[0])

    # 2. If dataset is empty, return majority class of original data
    elif len(data) == 0:
        return str(np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])
        ])

    # 3. If no features left, return majority class of parent
    elif len(features) == 0:
        return str(parent_node_class)

    # 4. Build tree
    else:
        # Set default value for this node
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
        ]
        
        # Calculate gains for all features
        gains = [info_gain(data, feature, target_attribute_name)[0] for feature in features]
        best_feature_index = np.argmax(gains)
        best_feature = features[best_feature_index]

        # Create tree structure
        tree = {best_feature: {}}

        # Remove best feature from features list
        remaining_features = [f for f in features if f != best_feature]

        # Add branches for each value of best feature
        all_values = np.unique(originaldata[best_feature])
        for value in all_values:
            sub_data = data[data[best_feature] == value]
            subtree = Id3(sub_data, originaldata, remaining_features, target_attribute_name, parent_node_class)
            tree[best_feature][str(value)] = subtree
            
        return tree

# --- Play Golf Dataset ---
def get_play_golf_data():
    return pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 'Sunny',
                    'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild',
                        'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High',
                     'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': ['False', 'False', 'True', 'False', 'False', 'True', 'True', 'False',
                  'False', 'False', 'True', 'True', 'False', 'True'],
        'PlayGolf': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
                     'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    })

# --- Routes ---

@app.route('/decision-tree')
def decision_tree_page():
    return render_template('decision_tree.html')

@app.route('/dt/dataset')
def dt_dataset():
    """Get Play Golf dataset"""
    try:
        df = get_play_golf_data()
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dt/calculate', methods=['POST'])
def dt_calculate():
    """
    Calculate Entropy and Info Gain for a specific feature step
    Used for educational purposes
    """
    try:
        data = request.get_json()
        target = data.get('target', 'PlayGolf')
        dataset_filters = data.get('filters', {}) # To filter data for subtree calc
        
        df = get_play_golf_data()
        
        # Apply filters (simulate being at a specific node)
        for col, val in dataset_filters.items():
            df = df[df[col] == val]
            
        if len(df) == 0:
            return jsonify({'entropy': 0, 'gains': {}})

        # Calculate current entropy
        current_entropy = entropy(df[target])
        
        # Calculate gain for each remaining attribute
        features = [col for col in df.columns if col != target and col not in dataset_filters]
        gains = {}
        
        for feature in features:
            gain, _, _ = info_gain(df, feature, target)
            gains[feature] = gain
            
        return jsonify({
            'total_samples': len(df),
            'current_entropy': current_entropy,
            'gains': gains,
            'features': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dt/build', methods=['GET'])
def dt_build():
    """Build the full tree and return structure"""
    try:
        df = get_play_golf_data()
        features = list(df.columns[:-1])
        tree_structure = Id3(df, df, features)
        return jsonify(tree_structure)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dt/predict', methods=['POST'])
def dt_predict():
    """Predict using the built tree"""
    try:
        data = request.get_json()
        df = get_play_golf_data()
        features = list(df.columns[:-1])
        
        # Build tree
        tree = Id3(df, df, features)
        
        # Traverse tree
        prediction = None
        current_node = tree
        path = []
        
        while isinstance(current_node, dict):
            # Get the feature to check (first key)
            feature = list(current_node.keys())[0]
            input_value = data.get(feature)
            path.append({'node': feature, 'value': input_value})
            
            # Navigate to next node
            if input_value in current_node[feature]:
                current_node = current_node[feature][input_value]
            else:
                # Value not in tree path
                return jsonify({'prediction': 'Unknown', 'path': path})
                
        prediction = current_node
        return jsonify({'prediction': prediction, 'path': path})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)