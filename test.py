import streamlit as st
from PIL import Image
import numpy as np
from Canny import Canny

uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

scanner = Canny()

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    if st.checkbox('Show original image'):
        st.image(image)
    if st.button('Extract information'):
        st.spinner('Extracting information')
        preprocess = scanner.preprocessing(image)
        final, result = scanner.scan(preprocess)
        st.image(result)
        st.success('Done!')