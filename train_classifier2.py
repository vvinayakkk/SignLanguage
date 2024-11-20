import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(data_path='./data.pickle', max_length=84):
    # Load data
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Pad or truncate data
    data = [
        np.pad(item, (0, max_length - len(item)), mode='constant') 
        if len(item) < max_length 
        else item[:max_length] 
        for item in data_dict['data']
    ]
    
    # Convert to numpy array
    data = np.array(data)
    labels = np.array(data_dict['labels'])
    
    # Encode labels
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    return data, labels_encoded, le

def train_models(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'Logistic Regression': LogisticRegression(multi_class='ovr', max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train and evaluate models
    model_results = {}
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Store results
        model_results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    return model_results

def main():
    # Load and prepare data
    X, y, label_encoder = load_and_prepare_data()
    
    # Train models
    model_results = train_models(X, y)
    
    # Save models and label encoder
    os.makedirs('models', exist_ok=True)
    
    # Save individual models
    for name, result in model_results.items():
        with open(f'models/{name.replace(" ", "_").lower()}_model.pkl', 'wb') as f:
            pickle.dump({
                'model': result['model'],
                'accuracy': result['accuracy'],
                'classification_report': result['classification_report']
            }, f)
    
    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Print model accuracies
    for name, result in model_results.items():
        print(f"{name} Accuracy: {result['accuracy']*100:.2f}%")

if __name__ == "__main__":
    main()