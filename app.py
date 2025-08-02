
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time

st.set_page_config(
    page_title="Object Detector - Duality AI Hackathon",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Duality AI - Space Station Object Detector")
st.markdown("Detects objects like **cheerios** and **soup** using a custom-trained YOLOv8 model.")

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train2/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting... please wait"):
        output_dir = "predicted"
        results = model.predict(source=image, save=True, save_txt=False, project=output_dir, name="results", exist_ok=True, conf=0.25)
        time.sleep(1)

    result_path = os.path.join(output_dir, "results", os.listdir(f"{output_dir}/results")[0])
    st.image(result_path, caption="✅ Detection Result", use_column_width=True)

    st.success("🎯 Detection Complete!")
    st.markdown("### 📊 Model Confidence Threshold: `0.25`\nCustomize in code if needed.")


st.markdown("---")
st.caption("🛰️ Built for the Duality AI Hackathon | Team Code Wizards 🚀")
=======
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time
import uuid

st.set_page_config(
    page_title="Object Detector - Duality AI Hackathon",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Duality AI - Space Station Object Detector")
st.markdown("Detects objects like **cheerios** and **soup** using a custom-trained **YOLOv8** model.")

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train4/weights/best.pt")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    run_id = str(uuid.uuid4())[:8]
    output_dir = "predicted"

    with st.spinner("🔍 Detecting... please wait..."):
        results = model.predict(
            source=image,
            save=True,
            save_txt=False,
            project=output_dir,
            name=run_id,
            exist_ok=True,
            conf=0.25
        )
        time.sleep(1)

    result_img_path = os.path.join(output_dir, run_id, os.listdir(f"{output_dir}/{run_id}")[0])
    st.image(result_img_path, caption="✅ Detection Result", use_column_width=True)

    boxes = results[0].boxes
    if boxes and boxes.cls.numel() > 0:
        class_names = model.names
        det_classes = [class_names[int(c)] for c in boxes.cls]
        det_conf = [round(float(c), 2) for c in boxes.conf]

        st.markdown("### 🔍 Detected Objects:")
        st.table({"Object": det_classes, "Confidence": det_conf})
    else:
        st.warning("⚠️ No objects detected. Try another image or lower confidence threshold.")

    # Download button
    with open(result_img_path, "rb") as file:
        st.download_button("⬇️ Download Result Image", file, file_name="detection_result.jpg", mime="image/jpeg")

    st.success("🎯 Detection Complete!")
    st.markdown("**Model Confidence Threshold:** `0.25` (You can modify this in the code)")

st.markdown("---")
st.caption("🛰️ Built for the Duality AI Hackathon | Team Code Wizards 🚀")
