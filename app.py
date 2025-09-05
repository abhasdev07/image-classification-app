import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
from rembg import remove
from colorthief import ColorThief
from expcolor import color_dict    
from PIL import Image
import requests
import time
from io import BytesIO
from streamlit_lottie import st_lottie
import io
import pyttsx3
import uuid

def generate_voice(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 125) 
    engine.setProperty('volume', 1.0)
    output_path = f"voice_{uuid.uuid4().hex[:8]}.mp3"
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    return output_path

DATASET_PATH  = r"D:\Dataset"      
IMG_SIZE      = 160                
MODEL_PATH    = "Model39.keras"     

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def get_color_name(hex_code):
    hex_code = hex_code.lower()
    if hex_code in color_dict:
        return color_dict[hex_code]
    r1, g1, b1 = tuple(int(hex_code.lstrip('#')[i:i+2],16) for i in (0,2,4))
    best, best_dist = None, None
    for h, name in color_dict.items():
        r2, g2, b2 = tuple(int(h.lstrip('#')[i:i+2],16) for i in (0,2,4))
        d = (r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2
        if best_dist is None or d < best_dist:
            best, best_dist = name, d
    return best

model = tf.keras.models.load_model(MODEL_PATH)

ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    batch_size=1,
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False
)
class_names = ds.class_names

def predict_image(uploaded_file):
    img_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    no_bg_bytes = remove(img_bytes, force_return_bytes=True)

    img = PILImage.open(io.BytesIO(no_bg_bytes)).convert("RGB")
    obj_img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = tf.keras.preprocessing.image.img_to_array(obj_img) / 255.0
    arr = np.expand_dims(arr, 0).astype("float16")
    preds = model.predict(arr)[0]
    obj_label = class_names[np.argmax(preds)]

    ct = ColorThief(io.BytesIO(no_bg_bytes))
    dom_rgb = ct.get_color(quality=10)
    hex_code = rgb_to_hex(dom_rgb)
    color_name = get_color_name(hex_code)

    return obj_label, color_name, dom_rgb

def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

def load_image_from_url(url: str):
    r = requests.get(url)
    return Image.open(BytesIO(r.content))

st.set_page_config(
    page_title="Kids Learning Toolkit",
    layout="wide"
)

if "page" not in st.session_state:
    st.session_state.page = 0
if "last_page" not in st.session_state:
    st.session_state.last_page = 0

def go_to(page_num):
    st.session_state.page = page_num

def restart():
    st.session_state.page = 0
    st.session_state.last_page = -1

with st.spinner(" Loading animations…"):
    lottie_learning = load_lottieurl("https://lottie.host/6c09c1eb-2c17-427c-a584-6236259308d3/amjEzGcvao.json")
    lottie_feedback = load_lottieurl("https://lottie.host/079c72cc-7459-4e1d-9e72-425ed48a260b/SE0FGNfr1F.json")

nav1, nav2, nav3 = st.columns([2,6,2])
with nav1:
    st.button(" Home", key="nav_home", use_container_width=True,
              on_click=go_to, args=(0,))
with nav2:
    st.button(" Learning Space", key="nav_learn", use_container_width=True,
              on_click=go_to, args=(1,))
with nav3:
    st.button(" Rate Experience", key="nav_rate", use_container_width=True,
              on_click=go_to, args=(2,))

if st.session_state.page != st.session_state.last_page:
    st.session_state.last_page = st.session_state.page

if st.session_state.page == 0:
    st.title("Welcome to Kids Learning Toolkit!")
    st.subheader("Smart learning starts with pictures")

    welcome_urls = [
        "https://i.pinimg.com/736x/12/90/46/1290461c5ea5cca6f8c8dfbe8106318b.jpg",
        "https://static.vecteezy.com/system/resources/previews/002/399/947/non_2x/happy-kids-studying-and-learning-vector.jpg",
        "https://cdn.firstcry.com/education/2022/10/19172448/Learning-English-Aphabet-for-Kids-Importance-and-Acticities.jpg",
        "https://www.greysprings.com/images/product_tiles/preschool_basics_tile.png",
        "https://kidstut.com/wp-content/uploads/2024/02/Slide1-min-2-1024x709.jpg",
        "https://www.shutterstock.com/image-illustration/kids-learning-fruits-name-helping-260nw-2420709085.jpg",
    ]

    for row in [welcome_urls[:3], welcome_urls[3:]]:
        cols = st.columns(3)
        for col, img_url in zip(cols, row):
            with st.spinner(" Loading image…"):
                try:
                    img = load_image_from_url(img_url)
                    img = img.resize((500, 500))
                    col.image(img)
                except:
                    col.warning(" Failed to load image")

    st.title("“Let’s keep exploring”")
    center = st.columns([3,1,3])[1]
    center.button("Next ", key="next_home", use_container_width=True,
                  on_click=go_to, args=(1,))

elif st.session_state.page == 1:
    st.title(" Learning Space")
    st_lottie(lottie_learning, height=300, key="learn_lottie")
    st.write("Let the kids explore, guess, and learn through images!")

    with st.sidebar:
        st.header(" How to Use")
        st.write("- Upload an image.")
        st.write("- Click **Show Answer** to find out.")
        st.write("- Click **Listen** to hear it aloud.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        if st.button("Show Answer ", use_container_width=True):
            with st.spinner("Analyzing..."):
                time.sleep(1)
                obj, color, rgb = predict_image(uploaded)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(uploaded, use_container_width=False, width=300)

            with col2:
                st.success(f"This is a **{obj}** and its color is **{color}**.")
                colorImage = PILImage.new("RGB", (150, 150), rgb)
                st.image(colorImage, caption=color, width=150)

                result_text = f"This is a {obj} and its color is {color}."
                audio_path = generate_voice(result_text)
                if audio_path:
                    with open(audio_path, 'rb') as f:
                        st.audio(f.read(), format='audio/mp3')  # or 'audio/wav'

    left, right = st.columns([1,1])
    left.button(" Back", key="back_learn", use_container_width=True,
                on_click=go_to, args=(0,))
    right.button("Next ", key="next_learn", use_container_width=True,
                 on_click=go_to, args=(2,))

elif st.session_state.page == 2:
    st.title(" Rate Your Experience")
    st_lottie(lottie_feedback, height=300, key="feedback_lottie")
    rating = st.slider("How much fun was this?", 1, 5, 4, key="rating_slider")

    if st.button("Submit Rating ", key="submit_rating", use_container_width=True):
        st.success(f"You rated us: {rating}/5")
        st.toast("Thanks for your feedback! ")
        st.write(" Thank You for exploring with us!")

    center = st.columns([3,1,3])[1]
    center.button("Restart ", key="restart", on_click=restart, use_container_width=True)
