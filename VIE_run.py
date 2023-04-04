import os
import random
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

from src.util.transforms import Transforms
from src.model.layoutlmv3 import LayoutLMv3

# UI
st.set_page_config(
    page_title='VIRIE',
    page_icon='💸',
)

st.header('Vietnamese Receipt Information Extraction')
st.sidebar.header('Tool Bar')

uploaded_file = st.file_uploader(
    "Choose an image", type=['png', 'jpg', 'jpeg'])


@st.cache_resource
def init_models():
    model_path = os.path.join(
        'src', 'model', 'models', 'segmentation_model.h5')
    model = load_model(model_path, compile=False)
    processor = Transforms(model)
    extractor = LayoutLMv3()
    return processor, extractor


processor, extractor = init_models()

fn = {
    'Warp': processor.warp_perspective,
    'Blur': processor.blur,
    'Contrast': processor.enhance_contrast,
    'Sharp': processor.enhance_sharp,
    'Binarize': processor.adaptive_binary_image,
    'Canny': processor.get_canny,
}

waiting = ['Từ từ thì cháo mới nhừ', "Đợi xíu má", "Waiting for you",
           "Người thành công là người kiên nhẫn, người không kiên nhẫn là thành thụ"]

options = st.sidebar.multiselect('Choose transformation', list(fn.keys()))
for option in options:
    if option == 'Blur':
        blur = st.sidebar.text_input(
            'Amount of blur', '0', help='The standard deviation value. The higher the standard deviation, the more blurred the image is')
    if option == 'Contrast':
        contrast = st.sidebar.text_input('Factor of contrasting', '1.5')
    if option == 'Binarize':
        mode = st.sidebar.selectbox('Mode', ['mean', 'gaussian'])
        block_size = st.sidebar.text_input('Block size', '11')
        constant = st.sidebar.text_input('Constant', '2')

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
                    if option == 'Warp':
                        bin_image = fn[option](bin_image)
                    elif option == 'Blur':
                        bin_image = fn[option](bin_image, float(blur))
                    elif option == 'Contrast':
                        bin_image = fn[option](bin_image, float(contrast))
                    elif option == 'Sharp':
                        bin_image = fn[option](bin_image)
                    elif option == 'Binarize':
                        bin_image = fn[option](
                            bin_image, mode, int(block_size), int(constant))
                    elif option == 'Canny':
                        bin_image = fn[option](bin_image)
            st.image(bin_image)
            st.text('Processed image')

            if st.button('Extract info'):
                result = extractor.predict(bin_image.convert('RGB'))

        if result:
            st.download_button(
                label="Download data as JSON",
                data=result,
                file_name='extracted_data.json',
                mime='application/json')
            st.json(result, expanded=False)

    elif process_button:
        st.warning('Chọn hình đi ba')
