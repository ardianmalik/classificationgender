import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Menghilangkan menu hamburger dan watermark Streamlit
hide_streamlit_style = """
<style>
# MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def get_face_box(net, frame, conf_threshold=0.7):
    """Mendeteksi wajah dalam gambar menggunakan model OpenCV DNN."""
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [
        104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 350)), 8)
    return opencv_dnn_frame, b_boxes_detect


# Judul aplikasi
st.write("""
# Prediksi Jenis Kelamin
""")

st.write("## Upload Foto Yang Memperlihatkan Wajah")

uploaded_file = st.file_uploader("Pilih file (JPG atau PNG):", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    cap = np.array(image)
    cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY))
    cap = cv2.imread('temp.jpg')

    # Model dan konfigurasi deteksi wajah serta jenis kelamin
    face_txt_path = "opencv_face_detector.pbtxt"
    face_model_path = "opencv_face_detector_uint8.pb"
    gender_txt_path = "gender_deploy.prototxt"
    gender_model_path = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    gender_classes = ['Gender: Pria', 'Gender: Wanita']

    # Load model deteksi wajah dan jenis kelamin
    gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

    # Perbesar area frame untuk memasukkan rambut atau bagian lain
    padding = 100  # Tambahkan lebih banyak ruang sekitar wajah
    frameFace, b_boxes = get_face_box(face_net, cap)

    if not b_boxes:
        st.write("Tidak ada wajah yang terdeteksi, silakan coba lagi dengan gambar lain.")

    for bbox in b_boxes:
        # Potong area wajah dengan padding
        face = cap[
            max(0, bbox[1] - padding): min(bbox[3] + padding, cap.shape[0] - 1),
            max(0, bbox[0] - padding): min(bbox[2] + padding, cap.shape[1] - 1)
        ]

        # Preprocess untuk model
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_pred_list = gender_net.forward()
        gender = gender_classes[gender_pred_list[0].argmax()]

        st.write(f"{gender}, Akurasi: {gender_pred_list[0].max() * 100:.2f}%")

        # Tampilkan label pada gambar asli
        label = "{}".format(gender)
        cv2.putText(
            frameFace,
            label,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Menampilkan gambar dengan bounding box dan label
    st.image(frameFace, channels="BGR")
