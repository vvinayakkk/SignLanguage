import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os

# Define the path to the model file
model_path = r"C:\Users\aasha\Downloads\my_model.keras"

# Verify file existence and accessibility
if not os.path.exists(model_path):
    st.error(f"Model file not found at: {model_path}")
    st.stop()

# Load your trained Keras model
try:
    cnn = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Mapping of class indices to alphabet letters
alphabet_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Function to make prediction
def predict_letter(image_path, model):
    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Make prediction
    result = model.predict(test_image)
    predicted_class_index = np.argmax(result)

    # Map class index to alphabet letter
    predicted_letter = alphabet_mapping.get(predicted_class_index, 'Unknown')
    
    return predicted_letter

# Streamlit app
def main():
    st.title("WELCOME TO SignSense")
    st.markdown("**Everyone's communication tool**!", unsafe_allow_html=True)
    st.markdown("<hr style='border: 2px solid black;'>", unsafe_allow_html=True)

        # Initial heading
    st.markdown(
        """
        <div style='text-align:center;'>
            <h2 style='color:#fb5a04;'>HOW YOU INTERACT MATTERS, DO IT TOGETHER WITH SIGNSENSE!!</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Sidebar with icons and dropdown
    st.sidebar.title("Menu")

    # Icon 1 with dropdown
    with st.sidebar.expander("Our Contribution"):
        st.write("Converting sign language to text fosters inclusivity by bridging communication gaps between the deaf and hearing communities. It enables better access to education, facilitates seamless communication in various settings, and empowers individuals to express themselves more effectively. Ultimately, it promotes a more equitable and diverse society by ensuring everyone can participate fully in social, educational, and professional spheres.")

        

    # Icon 2 with dropdown
    with st.sidebar.expander("Our Vision"):
        st.write("We want to bridge the gap of communication worldwide, thereby assisting every single person with deafness. In near future we plan to make our model ready for all the various sign languages around the world and also try to expand its use case opportunities. We will explore further opportunities in this project.")
        

    # Icon 3 with dropdown
    with st.sidebar.expander("Our Team"):
        st.write("Aashay Jadhav")
        st.write("Shubham Rajapurkar")
        st.write("Dhruv Kanadia")

    # List to store predicted alphabet letters
    predicted_letters = []

    # File uploader for multiple image inputs
    uploaded_files = st.file_uploader("Upload multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write("Number of uploaded files:", len(uploaded_files))  # Debug statement

        for uploaded_file in uploaded_files:
            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Save the uploaded image temporarily
            image_path = os.path.join('./temp', uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Make prediction for the uploaded image
            try:
                with st.spinner('Predicting...'):
                    predicted_letter = predict_letter(image_path, cnn)
                predicted_letters.append(predicted_letter)
                st.write(f"Predicted letter: {predicted_letter}")  # Debug statement

                # Display ASL sign for predicted letter (example)
                asl_images = {
                    'A': 'asl_a.jpg',
                    'B': 'asl_b.jpg',
                    # Add more mappings...
                }
                st.image(asl_images[predicted_letter], caption=f'ASL for {predicted_letter}', use_column_width=True)

            except Exception as e:
                st.error("")

        # Display the composed sentence or word
        if predicted_letters:
            composed_text = ''.join(predicted_letters)
            st.success(f"Predicted word: {composed_text}")

        # Feedback section
        feedback = st.text_input('Feedback:')
        if st.button('Submit Feedback'):
            # Process user feedback (you can add your feedback handling logic here)
            st.success('Thank you for your feedback!')

if __name__ == "__main__":
    main()
