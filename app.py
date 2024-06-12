import streamlit as st
import os
from utils import load_model, predict_digit
from PIL import Image

# Load the trained CNN model
model = load_model('mnist_cnn.pth')

def recognize_digit(image, model_type=1):
    digit = predict_digit(image, model, model_type=model_type)
    return digit

st.title("MNIST Digit Recognizer")

st.sidebar.title("Choose Input Method")
input_method = st.sidebar.radio("Select input method", ('Canvas', 'Upload from folder'))

if input_method == 'Canvas':
    from streamlit_drawable_canvas import st_canvas
    
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=5,
        stroke_color="black",
        background_color="#eee",
        update_streamlit=True,
        height=150,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        image = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
        st.image(image, caption="Drawn Image", use_column_width=True)
        digit = recognize_digit(image, model_type=1)
        st.success(f"Predicted Digit: {digit}")

else:
    # List all image files in the testSample folder
    image_files = os.listdir("testSample")

    # Display a selectbox for choosing an image from the folder
    selected_image = st.selectbox("Select an image:", image_files)

    if selected_image:
        image_path = os.path.join("testSample", selected_image)
        image = Image.open(image_path)
        st.image(image, caption="Selected Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        digit = recognize_digit(image, model_type=2)
        st.success(f"Predicted Digit: {digit}")

# Add developed by information
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Jayachandran P M")