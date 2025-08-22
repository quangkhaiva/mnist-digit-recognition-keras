import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = "models/mnist_tf_best.keras"

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

def center_pad_28x28(img_arr):
    """
    Nhận ảnh numpy 2D [0..255] dạng 'white digit on black background'
    -> crop theo bbox, resize giữ tỉ lệ về 20x20, pad về 28x28 và căn giữa.
    """
    ys, xs = np.where(img_arr > 10)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((28, 28), dtype=np.uint8)

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = img_arr[y0:y1+1, x0:x1+1]

    h, w = crop.shape
    if h > w:
        new_h, new_w = 20, int(round(20 * w / h))
    else:
        new_w, new_h = 20, int(round(20 * h / w))

    crop_img = Image.fromarray(crop)
    crop_img = crop_img.resize((max(1, new_w), max(1, new_h)), Image.BILINEAR)
    crop_rs = np.array(crop_img)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - crop_rs.shape[0]) // 2
    x_off = (28 - crop_rs.shape[1]) // 2
    canvas[y_off:y_off+crop_rs.shape[0], x_off:x_off+crop_rs.shape[1]] = crop_rs
    return canvas

def preprocess_pil(pil_img, invert=True, center=True):
    """
    Trả về tensor (1, 28, 28, 1) float32 [0..1] giống MNIST (white on black).
    - invert=True: dùng cho ảnh nền trắng chữ đen.
    - center=True : crop/resize/pad để căn giữa như MNIST.
    """
    img = pil_img.convert("L")
    if invert:
        img = ImageOps.invert(img)

    arr = np.array(img).astype(np.uint8)
    if center:
        arr = center_pad_28x28(arr)
    else:
        img = Image.fromarray(arr).resize((28, 28))
        arr = np.array(img)

    arr = arr.astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def predict_and_show(model, x):
    prob = model.predict(x, verbose=0)[0]
    pred = int(np.argmax(prob))
    st.markdown(f"### ✅ Dự đoán: **{pred}**")
    st.bar_chart(prob)

# ---------------- UI ----------------
st.set_page_config(page_title="MNIST Handwritten Digit Recognition", page_icon="📝")
st.title("📝 MNIST Handwritten Digit Recognition (TensorFlow/Keras)")

model, err = load_model()
if model is None:
    st.error(f"Không thể load model ở `{MODEL_PATH}`. Hãy train trước và đảm bảo file tồn tại.\n\nChi tiết lỗi: {err}")
    st.stop()

tab1, tab2 = st.tabs(["📁 Tải ảnh", "✍️ Vẽ trên canvas"])

with tab1:
    st.subheader("Tải ảnh PNG/JPG")
    invert = st.checkbox("Đảo màu (chọn nếu ảnh nền trắng, chữ đen)", value=True)
    center = st.checkbox("Căn giữa giống MNIST (khuyên dùng)", value=True)
    uploaded = st.file_uploader("Chọn ảnh (số 0–9)", type=["png", "jpg", "jpeg"])

    if uploaded:
        pil = Image.open(uploaded)
        st.image(pil, caption="Ảnh gốc", use_container_width=True)
        x = preprocess_pil(pil, invert=invert, center=center)
        show = (x[0, :, :, 0] * 255).astype(np.uint8)
        st.image(show, caption="Ảnh sau preprocess (28x28)", width=200)
        predict_and_show(model, x)

with tab2:
    st.subheader("Vẽ số 0–9")
    st.caption("Nét vẽ **màu trắng** trên **nền đen** (đúng định dạng MNIST).")
    stroke_w = st.slider("Độ dày nét", 8, 40, 20)

    canvas_res = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_w,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("🧠 Dự đoán từ canvas"):
        if canvas_res.image_data is None:
            st.warning("Hãy vẽ số trước đã nhé!")
        else:
            rgba = (canvas_res.image_data).astype(np.uint8)
            pil = Image.fromarray(rgba).convert("L")
            x = preprocess_pil(pil, invert=False, center=True)
            show = (x[0, :, :, 0] * 255).astype(np.uint8)
            st.image(show, caption="Ảnh sau preprocess (28x28)", width=200)
            predict_and_show(model, x)

st.caption("Model: `models/mnist_tf_best.keras`. Nếu thiếu file, hãy chạy script train để tạo.")
