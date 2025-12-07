import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter



os.makedirs("output/images", exist_ok= True)
os.makedirs("output/predictions", exist_ok= True)
os.makedirs("traning_data", exist_ok= True)

model = YOLO(r"C:\Users\adith\Documents\yolo\my_model\my_model.pt")

st.title("Egg Analysis Tool 😸")


st.header("Run the Analysis")

uploaded = st.file_uploader("Upload your image", type=["jpg","png","jpeg","tif"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(img)
    annotated = results[0].plot(line_width=2, font_size=20)

    object_count = len(results[0].boxes)
    st.write(f"**Objects detected:** {object_count}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    original_path = f"output/images/{ts}_original.jpg"
    annotated_path = f"output/images/{ts}_annotated.jpg"
    csv_path = f"output/predictions/{ts}.csv"

    cv2.imwrite(original_path, img)
    cv2.imwrite(annotated_path, annotated)

    detections = results[0].boxes

    labels = [model.names[int(box.cls)] for box in detections]
    confidences = [float(box.conf) for box in detections]

    class_counts = Counter(labels)


    
    df = pd.DataFrame({
    "Object_Name": labels,
    "Confidence(%)": [round(c*100,2) for c in confidences]
    })
    df["Total_Objects_in_this_class"] = df["Object_Name"].map(class_counts)
    df.to_csv(csv_path, index=False)

    tab1, tab2 = st.tabs(["Summary", "Graphs"])

    
    with tab1:
        st.subheader("🧾 Object Summary")
        for obj, count in class_counts.items():
            st.write(f"**{obj} : {count} detected**")

        st.subheader("Detection Details")
        st.dataframe(df)
    

        st.image(annotated, channels="BGR")

    with tab2:

    #BAR GRAPH
        st.subheader("Class Count Bar Graph 🤓")

        fig, ax = plt.subplots(figsize=(6,4), dpi=120)
        
        fig.patch.set_facecolor("#9B9B9B")
        ax.set_facecolor("#9B9B9B")
        bars = ax.bar(class_counts.keys(), class_counts.values(), color='maroon')
        ax.set_xlabel("Object Class")
        ax.set_ylabel("Count")
        ax.set_title("Detected Objects per Class")
        ax.grid(alpha=0.3)

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{bar.get_height()}",
                ha='center', va='bottom', fontsize=10
            )

        st.pyplot(fig)

        #Confidence graph
        st.subheader("Confidence Score Distribution 😎")

        fig2, ax2 = plt.subplots(figsize=(6,4), dpi=120)
        fig.patch.set_facecolor("#9B9B9B")
        ax2.hist(df["Confidence(%)"], bins=10, edgecolor='black', color='pink',  alpha=0.8)
        ax2.set_xlabel("Confidence (%)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Confidence Distribution")
        ax2.grid(alpha=0.3)

        st.pyplot(fig2)

        #Pie chart
        st.subheader("Class Distribution Pie Chart 🧐")

        fig3, ax3 = plt.subplots()
        ax3.pie(class_counts.values(), labels=class_counts.keys(), autopct="%1.1f%%")
        ax3.set_title("Class Distribution")

        st.pyplot(fig3)

    st.success("Detection complete — files saved locally!")


    st.download_button(
        label="Download CSV 🪱",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"prediction_{ts}.csv",
        mime="text/csv"
    )







st.title("Upload Images for Future Training")
st.header("No detection will be performed on these !")
training_uploads = st.file_uploader(
    "Upload raw images to store for dataset creation",
    type=["jpg", "jpeg", "png","tif", "mp4"],
    accept_multiple_files=True
)

if training_uploads:
    saved_files = []

    for file in training_uploads:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        save_path = f"traning_data/training_data{ts}_{file.name.replace(' ', '_')}"
        bytes_data = file.read()

        with open(save_path, "wb") as f:
            f.write(bytes_data)

        saved_files.append(save_path)

    st.success(f"✅ Saved {len(saved_files)} images for training!")

    with st.expander("Show Saved File Paths"):
        for path in saved_files:
            st.write(path)
    
