import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
import os

# create directory for uploads
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# load the pre-trained Inception V3 model
model = tf.keras.models.load_model("inceptionV3_model.h5")


# preprocess image for InceptionV3
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (299, 299))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# decode the predictions
def decode_predictions(preds):
    decode_preds = tf.keras.applications.inception_v3.decode_predictions(preds, top=3)[0]
    return decode_preds

# set up title
st.title("Video Object Detection")

# upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

# search query
search_query = st.text_input("Enter the object to search for: ")


# check if the video has been uploaded
if st.button("Upload"):
    if uploaded_file is not None:
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.video(save_path)
        
        print("File Upload Sucessfully")
        # open the video using OpenCV
        video = cv2.VideoCapture(save_path)
        
        # store the frames in a folder
        # get the labels of the frames
        # output the label with the respective search object
        
        # iterate through video frames
        frame_count = 0
        frames_with_object = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            # preprocess using InceptionV3
            preprocessed_frame = preprocess_image(frame_rgb)
            for i in range(len(preprocessed_frame)):
                print(i)
            
            # predict using InceptionV3
            preds = model.predict(preprocessed_frame)
            decoded_preds = decode_predictions(preds)
            
            # check id search query matches any detected objects
            for _,label,_ in decoded_preds:
                if search_query.lower() in label.lower():
                    frames_with_object.append(frame)
                    break
                    
            frame_count += 1
        video.release()
        
        # diplay results
        if frames_with_object:
            st.success(f"Found '{search_query}' in the video!")
            for i, frame in enumerate(frames_with_object):
                st.image(frame, caption=f"Frame {i+1}")
        else:
            st.error("Object doesn't exist!!!")
else:
    st.warning("Please upload a video file and enter a search query.")
