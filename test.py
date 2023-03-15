from Canny import Canny
from Invoice_extraction import InvoiceExtraction

import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import random
from inference import LayoutLMv3

# setup
scanner = Canny()
model_path = r'Invoice_Segmentation_model.h5'
model = load_model(model_path, compile=False)
extractor = InvoiceExtraction(model)
extract_model = LayoutLMv3()

fn = {'Warp': extractor.warp_perspective,
      'Blur': extractor.blur,
      'Contrast': extractor.enhance_contrast,
      'Sharp': extractor.enhance_sharp,
      'Binarize': extractor.adaptive_binary_image
      }
waiting = ['Từ từ thì cháo mới nhừ', "Đợi xíu má", "Waiting for you",
           "Người thành công là người kiên nhẫn, người không kiên nhẫn là thành thụ"]

# UI
st.set_page_config(
    page_title='Bill Information Extractor',
    page_icon='💸',
)

st.header('Project demo')
st.sidebar.header('Tool Bar')

uploaded_file = st.file_uploader(
    "Choose an image", type=['png', 'jpg', 'jpeg'])

options = st.sidebar.multiselect('Choose transformation', list(fn.keys()))
process_button = st.sidebar.button('Process')

if uploaded_file:
    image = Image.open(uploaded_file)
    bin_image = image.convert('L')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image)
        st.text('Original image')

    with col2:
        with st.spinner(random.choice(waiting)):
            if process_button:
                for option in options:
                    bin_image = fn[option](bin_image)
            st.image(bin_image)
            if process_button:
                st.text('Processed image')
            if st.button('Extract info'):
                result = extract_model.predict(bin_image)
                print(result)
                # st.text(result)

elif process_button:
    st.warning('Chọn hình đi ba')
