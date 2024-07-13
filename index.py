import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Mengatur konfigurasi halaman
st.set_page_config(page_title="Image Processing App", page_icon=":camera:", layout="centered")
st.title("Streamlit Image Processing Challenge SIC 5")
st.write("""
### Unggah gambar Anda dan pilih efek pengolahan gambar:
""")

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])
model = YOLO('yolov8n.pt')
object_names = list(model.names.values())

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    img = img[:, :, ::-1].copy()
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Deteksi wajah
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Deteksi objek dengan YOLO
    results = model(img)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = object_names[int(box.cls[0])]
            confidence = box.conf[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    st.image(Image.fromarray(image), caption="Gambar yang diunggah", use_column_width=True)

    # Menampilkan informasi gambar
    st.subheader("Informasi Gambar")
    st.write(f"Ukuran gambar: {Image.fromarray(image).width} x {Image.fromarray(image).height}")

    # Menampilkan histogram warna
    st.subheader("Histogram Warna")
    color = ('r', 'g', 'b')
    fig, ax = plt.subplots()
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(histr, color=col)
        ax.set_xlim([0, 256])
    st.pyplot(fig)

    # Menambahkan opsi pengolahan gambar sederhana
    st.subheader("Pengolahan Gambar")
    option = st.selectbox("Pilih efek gambar", ["None", "Grayscale", "Invert", "Mirror", "Gaussian Blur", "Threshold"])

    # Mengolah gambar berdasarkan pilihan pengguna
    if option == "Grayscale":
        processed_image = ImageOps.grayscale(Image.fromarray(image))
        st.image(processed_image, caption="Gambar Grayscale", use_column_width=True)
    elif option == "Invert":
        processed_image = ImageOps.invert(Image.fromarray(image).convert("RGB"))
        st.image(processed_image, caption="Gambar Invert", use_column_width=True)
    elif option == "Mirror":
        processed_image = ImageOps.mirror(Image.fromarray(image))
        st.image(processed_image, caption="Gambar Mirror", use_column_width=True)
    elif option == "Gaussian Blur":
        processed_image = Image.fromarray(cv2.GaussianBlur(image, (7, 7), 0))
        st.image(processed_image, caption="Gambar Blur", use_column_width=True)
    elif option == "Threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        processed_image = Image.fromarray(thresh)
        st.image(processed_image, caption="Gambar Threshold", use_column_width=True)
    else:
        st.image(image, caption="Gambar Asli", use_column_width=True)
else:
    st.write("Silakan unggah gambar untuk memulai.")
