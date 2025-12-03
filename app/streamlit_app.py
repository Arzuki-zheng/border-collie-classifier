import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

st.title("邊境牧羊犬辨識器")
@st.cache_resource
def get_model():
    # 用新的 .keras 檔
    return load_model("models/border_collie_model.keras")
model = get_model()

def preprocess(img):
    img = img.resize((224,224))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, 0)

file = st.file_uploader("上傳一張狗狗照片", type=["jpg","jpeg","png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="輸入影像", width=320)
    prob = float(model.predict(preprocess(img))[0][0])
    label = "邊境牧羊犬" if prob<0.5 else "其他狗"
    st.success(f"結果：{label}（邊境牧羊犬機率估計={1-prob:.3f}）")
