from Canny import Canny
from Invoice_extraction import InvoiceExtraction

import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import random
from inference import LayoutLMv3

# UI
st.set_page_config(
    page_title='Bill Information Extractor',
    page_icon='💸',
)

st.header('Project demo')
st.sidebar.header('Tool Bar')

uploaded_file = st.file_uploader(
    "Choose an image", type=['png', 'jpg', 'jpeg'])


@st.cache_resource
def init_models():
    model_path = r'Invoice_Segmentation_model.h5'
    model = load_model(model_path, compile=False)
    processor = InvoiceExtraction(model)
    extractor = LayoutLMv3()
    return processor, extractor


processor, extractor = init_models()

fn = {'Warp': processor.warp_perspective,
      'Blur': processor.blur,
      'Contrast': processor.enhance_contrast,
      'Sharp': processor.enhance_sharp,
      'Binarize': processor.adaptive_binary_image,
      'Canny': processor.get_canny,
      }

waiting = ['Từ từ thì cháo mới nhừ', "Đợi xíu má", "Waiting for you",
           "Người thành công là người kiên nhẫn, người không kiên nhẫn là thành thụ"]


options = st.sidebar.multiselect('Choose transformation', list(fn.keys()))
process_button = st.sidebar.button('Process')

result = None
with st.spinner(random.choice(waiting)):
    if uploaded_file:
        image = Image.open(uploaded_file)
        bin_image = image.convert('L')
        col1, col2 = st.columns(2)
        with col1:
            st.image(image)
            st.text('Original image')

        with col2:

            if process_button:
                bin_image = image.convert('L')
                for option in options:
                    bin_image = fn[option](bin_image)
            st.image(bin_image)
            if process_button:
                st.text('Processed image')
            if st.button('Extract info'):
                result = extractor.predict(bin_image.convert('RGB'))

        if result:
            st.json(result)
            st.download_button(
                label="Download data as JSON",
                data=result,
                file_name='extracted_data.json',
                mime='text/csv',
            )

    elif process_button:
        st.warning('Chọn hình đi ba')
