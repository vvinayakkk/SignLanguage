import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import spacy
from transformers import pipeline

class AdvancedSignLanguageDetector:
    def __init__(self):
        # Load models
        self.models = self.load_models()
        
        # Load label encoder
        with open('models/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
        
        # NLP models
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except:
            st.warning("Some NLP features may not be available. Please install required models.")
            self.nlp = None
            self.sentiment_analyzer = None
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
                            12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
                            23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7',
                            34: '8', 35: '9', 36: 'I Love You', 37: 'EAT', 38: 'OK', 39: 'HELP', 40: 'Bus', 41: 'Calm Down', 42: 'Church', 
                            43: 'Cover', 44: 'Family', 45: 'Father', 46: 'Fine', 47: 'Goodbye', 48: 'Hate You', 49: 'Help', 50: 'Home', 
                            51: 'Hungry', 52: 'I am I', 53: 'Key', 54: 'Lock', 55: 'Mother', 56: 'No', 57: 'Sorry', 58: 'Stand', 
                            59: 'Stop', 60: 'Taxi', 61: 'Telephone', 62: 'Water', 63: 'Where', 64: 'Why', 65: 'Ship', 66: 'Airplane', 
                            67: 'Car', 68: 'Okay', 69: 'Yes', 70: 'Help', 71: 'help'}
    def load_models(self):
        model_names = [
            'random_forest_model.pkl', 
            'gradient_boosting_model.pkl', 
            'svm_(rbf)_model.pkl',
            'neural_network_model.pkl',
            'logistic_regression_model.pkl',
            'decision_tree_model.pkl',
            'extra_trees_model.pkl',
            'gaussian_naive_bayes_model.pkl',
            'k-nearest_neighbors_model.pkl'
        ]
        
        models = {}
        for name in model_names:
            try:
                with open(f'models/{name}', 'rb') as f:
                    model_data = pickle.load(f)
                    models[name.replace('_model.pkl', '').replace('_', ' ').title()] = model_data['model']
            except Exception as e:
                st.error(f"Could not load model {name}: {e}")
        
        return models

    def predict_with_all_models(self, data_aux):
        # Ensure data is correctly formatted
        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))
        elif len(data_aux) > 84:
            data_aux = data_aux[:84]
        
        # Predict with each model and return percentage of predictions
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict([np.asarray(data_aux)])
            pred_label_index = pred[0]
            pred_label = self.labels_dict.get(pred_label_index, "Unknown")
            pred_prob = model.predict_proba([np.asarray(data_aux)])[0]
            pred_percent = max(pred_prob) * 100  # Percentage of the most likely class
            predictions[name] = {
                'label': pred_label,
                'confidence': round(pred_percent, 2)
            }
        
        return predictions

    def advanced_nlp_analysis(self, predictions):
        if not self.nlp or not self.sentiment_analyzer:
            return {}
        
        # Combine predictions into a sentence
        combined_text = " ".join([f"{key}: {value['label']}" for key, value in predictions.items()])
        
        # Spacy NLP analysis
        doc = self.nlp(combined_text)
        
        # Basic NLP features
        nlp_analysis = {
            'Named Entities': [ent.text for ent in doc.ents],
            'Parts of Speech': [token.pos_ for token in doc],
            'Dependencies': [token.dep_ for token in doc]
        }
        
        # Sentiment analysis
        try:
            sentiment = self.sentiment_analyzer(combined_text)[0]
            nlp_analysis['Sentiment'] = sentiment
        except:
            nlp_analysis['Sentiment'] = "Analysis not available"
        
        return nlp_analysis

def main():
    st.set_page_config(layout="wide", page_title="Advanced Sign Language Detector")
    
    # Initialize detector
    detector = AdvancedSignLanguageDetector()
    
    # Session state for predictions
    if 'predictions_history' not in st.session_state:
        st.session_state.predictions_history = []
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a Page", 
        ["Real-Time Detection", "Model Comparison", "NLP Insights"])
    
    if page == "Real-Time Detection":
        st.title("🖐️ Multi-Model Sign Language Detection")
        
        # Camera feed
        run = st.checkbox("Start Detection")
        col1, col2 = st.columns(2)
        
        with col1:
            stframe = st.empty()
        
        with col2:
            st.subheader("Model Predictions")
            prediction_display = st.empty()
            nlp_display = st.empty()
            detected_text_display = st.empty()
        
        # Prediction history
        history_container = st.container()
        
        if run:
            cap = cv2.VideoCapture(0)
            
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
                        
                        # Predict using all models
                        predictions = detector.predict_with_all_models(data_aux)
                        
                        # NLP Analysis
                        nlp_analysis = detector.advanced_nlp_analysis(predictions)
                        
                        # Draw predictions on frame
                        for name, pred in predictions.items():
                            cv2.putText(frame, f"{name[:10]}: {pred['label']} ({pred['confidence']}%)", 
                                        (10, 30 + list(predictions.keys()).index(name)*30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        # Update prediction display
                        prediction_display.table(pd.DataFrame.from_dict(predictions, orient='index', columns=['Label', 'Confidence (%)']))
                        nlp_display.json(nlp_analysis)
                        detected_text_display.text(f"Detected Text: {' '.join([f'{key}: {value['label']}' for key, value in predictions.items()])}")
                        
                        # Store predictions in history
                        st.session_state.predictions_history.append(predictions)
                
                # Display frame
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            cap.release()
        
        # Prediction History
        with history_container:
            st.subheader("Prediction History")
            history_df = pd.DataFrame(st.session_state.predictions_history)
            st.dataframe(history_df)
    
    elif page == "Model Comparison":
        st.title("🧠 Model Performance Comparison")
        
        # Comparison chart
        model_names = list(detector.models.keys())
        accuracy_data = [np.random.rand() for _ in model_names]  # Replace with actual model accuracy values
        
        fig = go.Figure([go.Bar(x=model_names, y=accuracy_data)])
        fig.update_layout(title="Model Performance Comparison", xaxis_title="Model", yaxis_title="Accuracy")
        st.plotly_chart(fig)
    
    elif page == "NLP Insights":
        st.title("🔍 Advanced NLP Analysis")
        # Placeholder for displaying NLP analysis results, currently shows a sample JSON from the last prediction
        if st.session_state.predictions_history:
            latest_predictions = st.session_state.predictions_history[-1]
            nlp_analysis = detector.advanced_nlp_analysis(latest_predictions)
            st.json(nlp_analysis)

if __name__ == "__main__":
    main()
