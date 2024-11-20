import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class SignLanguageDetector:
    def __init__(self):
        # Load pre-trained models
        self.rf_model = self.load_model('./model.p')
        self.additional_models = {
            'Neural Network': self.train_neural_network(),
            'Support Vector Machine': self.train_svm()
        }
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
        
        # Labels dictionary
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
               23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7',
               34: '8', 35: '9', 36: 'I Love You', 37: 'EAT', 38: 'OK', 39: 'HELP',40: 'Bus', 41: 'Calm Down', 42: 'Church', 43: 'Cover', 44: 'Family', 45: 'Father', 
    46: 'Fine', 47: 'Goodbye', 48: 'Hate You', 49: 'Help', 50: 'Home', 51: 'Hungry', 
    52: 'I am I', 53: 'Key', 54: 'Lock', 55: 'Mother', 56: 'No', 57: 'Sorry', 
    58: 'Stand', 59: 'Stop', 60: 'Taxi', 61: 'Telephone', 62: 'Water', 63: 'Where', 
    64: 'Why', 65: 'Ship',66: 'Airplane',67: 'Car',68: 'Okay',69: 'Yes',70: 'Help',71: 'help'}

    def load_model(self, path):
        model_dict = pickle.load(open(path, 'rb'))
        return model_dict['model']

    def train_neural_network(self):
        # Load data
        data_dict = pickle.load(open('./data.pickle', 'rb'))
        max_length = 84
        data = [np.pad(item, (0, max_length - len(item))) if len(item) < max_length else item[:max_length] for item in data_dict['data']]
        data = np.asarray(data)
        labels = np.asarray(data_dict['labels'])
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        
        # Train Neural Network
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(x_train, y_train)
        return mlp

    def train_svm(self):
        # Load data
        data_dict = pickle.load(open('./data.pickle', 'rb'))
        max_length = 84
        data = [np.pad(item, (0, max_length - len(item))) if len(item) < max_length else item[:max_length] for item in data_dict['data']]
        data = np.asarray(data)
        labels = np.asarray(data_dict['labels'])
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        
        # Train SVM
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(x_train, y_train)
        return svm

    def predict(self, model, data_aux):
        # Ensure data is correctly formatted
        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]
        
        prediction = model.predict([np.asarray(data_aux)])
        return self.labels_dict[int(prediction[0])]

    def generate_model_comparison(self):
        # Load data
        data_dict = pickle.load(open('./data.pickle', 'rb'))
        max_length = 84
        data = [np.pad(item, (0, max_length - len(item))) if len(item) < max_length else item[:max_length] for item in data_dict['data']]
        data = np.asarray(data)
        labels = np.asarray(data_dict['labels'])
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
        
        # Prepare results dictionary
        model_results = {}
        models = {
            'Random Forest': self.rf_model,
            'Neural Network': self.additional_models['Neural Network'],
            'Support Vector Machine': self.additional_models['Support Vector Machine']
        }
        
        for name, model in models.items():
            y_pred = model.predict(x_test)
            model_results[name] = {
                'accuracy': classification_report(y_test, y_pred, output_dict=True)['accuracy'],
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        
        return model_results

def main():
    st.set_page_config(layout="wide", page_title="Advanced Sign Language Detector")
    
    # Initialize detector
    detector = SignLanguageDetector()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a Page", 
        ["Real-Time Detection", "Model Comparison", "Data Insights", "About Project"])
    
    if page == "Real-Time Detection":
        st.title("🖐️ Real-Time Sign Language Detection")
        
        # Model selection
        model_choice = st.selectbox("Select Detection Model", 
            ["Random Forest", "Neural Network", "Support Vector Machine"])
        
        # Camera feed
        run = st.checkbox("Run Detection")
        stframe = st.empty()
        
        if run:
            cap = cv2.VideoCapture(0)
            
            # Select appropriate model
            selected_model = {
                "Random Forest": detector.rf_model,
                "Neural Network": detector.additional_models['Neural Network'],
                "Support Vector Machine": detector.additional_models['Support Vector Machine']
            }[model_choice]
            
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.hands.process(frame_rgb)
                
                data_aux = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        detector.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, 
                            detector.mp_hands.HAND_CONNECTIONS,
                            detector.mp_drawing_styles.get_default_hand_landmarks_style(),
                            detector.mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        x_ = [landmark.x for landmark in hand_landmarks.landmark]
                        y_ = [landmark.y for landmark in hand_landmarks.landmark]
                        
                        # Prepare data for prediction
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                            data_aux.append(hand_landmarks.landmark[i].y - min(y_))
                        
                        H, W, _ = frame.shape
                        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
                        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
                        
                        # Predict using selected model
                        predicted_character = detector.predict(selected_model, data_aux)
                        
                        # Draw prediction
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            cap.release()
    
    elif page == "Model Comparison":
        st.title("🧠 Model Performance Comparison")
        
        # Generate model comparison
        model_results = detector.generate_model_comparison()
        
        # Accuracy Comparison
        st.subheader("Model Accuracy Comparison")
        accuracies = {name: results['accuracy'] * 100 for name, results in model_results.items()}
        fig = px.bar(x=list(accuracies.keys()), y=list(accuracies.values()), 
                     title="Model Accuracy Comparison",
                     labels={'x': 'Model', 'y': 'Accuracy (%)'})
        st.plotly_chart(fig)
        
        # Detailed Metrics
        st.subheader("Detailed Performance Metrics")
        for name, results in model_results.items():
            st.write(f"### {name} Model")
            st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
    
    elif page == "Data Insights":
        st.title("📊 Sign Language Dataset Insights")
        # Load dataset
        data_dict = pickle.load(open('./data.pickle', 'rb'))
        labels = data_dict['labels']
        
        # Label distribution
        st.subheader("Label Distribution")
        label_counts = pd.Series(labels).value_counts()
        fig = px.pie(values=label_counts.values, names=label_counts.index, 
                     title="Distribution of Sign Language Gestures")
        st.plotly_chart(fig)
    
    elif page == "About Project":
        st.title("🤲 Advanced Sign Language Detection")
        st.markdown("""
        ### Project Overview
        - **Objective**: Real-time sign language gesture recognition
        - **Technologies**: 
            - MediaPipe for hand landmark detection
            - Multiple machine learning models
            - Streamlit for interactive visualization
        
        ### Key Features:
        - Real-time hand gesture detection
        - Multi-model comparison
        - Interactive data insights
        """)

if __name__ == "__main__":
    main()