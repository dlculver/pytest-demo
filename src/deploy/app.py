import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import sys
import os
import numpy as np

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src.deploy.utils import load_model, preprocess_image, get_processed_image_for_display

def main():
    st.title("MNIST Digit Recognition")
    st.markdown("Draw a digit (0-9) on the canvas below, then click 'Predict' to see the result. Use the canvas toolbar to clear or adjust your drawing.")

    # Canvas settings
    canvas_size = 280
    stroke_width = 20

    # Draw canvas with toolbar enabled
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",         # White strokes
        background_color="#000000",     # Black background
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        display_toolbar=True,           # Enable toolbar with clear, undo, etc.
        update_streamlit=True,          # Update on each stroke (optional, for responsiveness)
        key="canvas",
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("mnist_cnn.pth").to(device)

    # Predict button
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            # Preprocess the drawn image
            img_tensor = preprocess_image(canvas_result.image_data)
            img_tensor = img_tensor.to(device)

            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                prediction = torch.argmax(output, dim=1).item()
                probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

            # Display results
            st.subheader("Prediction Results")
            st.write(f"Predicted Digit: **{prediction}**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Class Probabilities:")
                for i, prob in enumerate(probabilities):
                    st.write(f"Digit {i}: {prob:.4f}")
            with col2:
                # Show processed 28x28 image for debugging
                processed_img = get_processed_image_for_display(canvas_result.image_data)
                st.image(processed_img, caption="Processed 28x28 Input", width=150)
        else:
            st.warning("Please draw something on the canvas first!")

if __name__ == "__main__":
    main()