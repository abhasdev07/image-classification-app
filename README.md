# image-classification-app
An interactive AI-powered learning tool for kids that identifies everyday objects and their colors from images. Built with Streamlit, it removes image backgrounds, detects dominant colors, classifies objects using MobileNetV2, and reads the result aloud using text-to-speech.

**Features**
- Image input 
- Automatic background removal from uploaded images
- Detection of the dominant color of the object
- Object classification using a pre-trained MobileNetV2 model
- Audio output of the object name and color using text-to-speech

  **Technology Used**
- Frontend: Streamlit
- Backend: Python
- Machine Learning: TensorFlow, Keras (MobileNetV2)
- Libraries:
  - rembg (background removal)
  - colorthief (color extraction)
  - pyttsx3 (text-to-speech)

 **Project Structure**
 
├── app.py              # Main Streamlit app to run the UI

├── expcolor.py         # Color extraction logic using ColorThief

├── finalOutcome.ipynb  # Jupyter Notebook for combining model output and display

├── model39.keras       # Pre-trained MobileNetV2 model file

├── README.md           # Project documentation
