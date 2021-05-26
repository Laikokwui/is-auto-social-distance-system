import streamlit as st
from detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import datetime
import wget
from PIL import Image


#DB
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB Functions for Video
def create_video_table():
    c.execute('CREATE TABLE IF NOT EXISTS videotable(author TEXT,title TEXT,file_date DATE,path TEXT)')

def add_video(author,title,file_date,path):
    c.execute('INSERT INTO videotable(author,title,file_date,path) VALUES (?,?,?,?)', (author,title,file_date,path))
    conn.commit()

def view_all_videos():
    c.execute("SELECT * FROM videotable")
    data = c.fetchall()
    return data

def view_by_video_author():
    c.execute('SELECT DISTINCT author FROM videotable')
    data = c.fetchall()
    return data

def get_path_by_video_author(author):
    c.execute('SELECT path FROM videotable WHERE author="{}"'.format(author))
    data = c.fetchall()
    return data

# DB Functions for Image
def create_image_table():
    c.execute('CREATE TABLE IF NOT EXISTS imagetable(author TEXT,title TEXT,file_date DATE,path TEXT)')

def add_image(author,title,file_date,path):
    c.execute('INSERT INTO imagetable(author,title,file_date,path) VALUES (?,?,?,?)', (author,title,file_date,path))
    conn.commit()

def view_all_images():
    c.execute("SELECT * FROM imagetable")
    data = c.fetchall()
    return data

def view_by_image_author():
    c.execute('SELECT DISTINCT author FROM imagetable')
    data = c.fetchall()
    return data

def get_path_by_image_author(author):
    c.execute('SELECT path FROM imagetable WHERE author="{}"'.format(author))
    data = c.fetchall()
    return data


# save & upload helper function
def save_uploaded_image(uploaded_image):
    with open(os.path.join("images",uploaded_image.name),"wb") as f:
        f.write(uploaded_image.getbuffer())

def save_uploaded_video(uploaded_video):
    with open(os.path.join("videos",uploaded_video.name),"wb") as f:
        f.write(uploaded_video.getbuffer())


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img



def main():

    sidebar = st.sidebar.selectbox('Choose one of the following', ('Welcome', 'Add an Image','Add a Video','View All File','Image Analysis', 'Real Time Video Analysis'))

    if sidebar == 'Welcome':
        welcome()
    if sidebar == 'Add an Image':
        add_an_image()
    if sidebar == 'Add a Video':
        add_a_video()
    if sidebar == 'View All File':
        view_all_file()
    if sidebar == 'Image Analysis':
        image_analysis()
    if sidebar == 'Real Time Video Analysis':
        video_analysis()


def welcome():
    st.title("Automated Social Distancing Monitoring System by AI4Life")
    st.subheader('Team members:')
    st.text('1. Lai Kok Wui (101211447)\n2. Lee Zhe Sheng (10215371)\n3. Didier Luther Ho Chih-Yuan (101214093)\n4. Abraham Tan Chiun Wu (101213825)')

    st.subheader('A0 Poster for AI4LIFE video presentation')
    st.image('images/poster.jpg',use_column_width=True)

def add_an_image():
    st.title("Upload an image")
    create_image_table()
    file_author = st.text_input("Enter your name", max_chars=50)
    file_title = st.text_input("Enter Desire File Name")
    file_date = st.date_input("Created Date")
    image_file = st.file_uploader("Upload An Image",type=['png', 'jpg', 'jpeg'])
    if image_file is not None:
        file_details = {"FileName":image_file.name,"FileType":image_file.type}
        # st.write(file_details)
        # st.write(type(image_file))
        img = load_image(image_file)
        st.image(img)
        path = os.path.join("images", image_file.name)
        save_uploaded_image(image_file)
    if st.button("Add"):
        add_image(file_author, file_title, file_date, path)
        st.success("File: {} saved".format(file_title))

def add_a_video():
    st.title("Upload a video")
    create_video_table()
    file_author = st.text_input("Enter your name", max_chars=50)
    file_title = st.text_input("Enter Desire File Name")
    file_date = st.date_input("Created Date")
    video_file = st.file_uploader("Upload An Image",type=['mp4'])
    if video_file is not None:
        file_details = {"FileName":video_file.name,"FileType":video_file.type}
        # st.write(file_details)
        # st.write(type(video_file))
        img = video_file.read()
        st.video(img)
        path = os.path.join("videos", video_file.name)
        save_uploaded_video(video_file)
    if st.button("Add"):
        add_video(file_author, file_title, file_date, path)
        st.success("File: {} saved".format(file_title))

def view_all_file():
    st.header("View All Files")
    images = view_all_images()
    videos = view_all_videos()
    st.subheader("Image Database")
    st.write(images)
    st.subheader("Video Database")
    st.write(videos)

def image_analysis():
    st.title('Real Time Social Distancing Monitor System with Image')
    cuda = st.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

    st.subheader('Test Demo image')
    all_titles = [i[0] for i in view_by_image_author()]
    option = st.selectbox('Your Name', all_titles)
    all_path = [i[0] for i in get_path_by_image_author(option)]
    option2 = st.selectbox("Select your uploaded file", all_path)


    USE_GPU = bool(cuda)
    MIN_DISTANCE = 50


    labelsPath = "yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = "yolo-coco/yolov4.weights"
    configPath = "yolo-coco/yolov4.cfg"


    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    if USE_GPU:

        st.info("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    if st.button('Start'):

        st.info("[INFO] loading YOLO from disk...")
        st.info("[INFO] accessing image stream...")

        for i in view_by_image_author():
            if option == i[0]:
                vs = cv2.VideoCapture(option2)
            else:
                vs = cv2.VideoCapture(0)
            writer = None
            image_placeholder = st.empty()

        while True:

            (grabbed, frame) = vs.read()

            if not grabbed:
                break

            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                                    personIdx=LABELS.index("person"))

            violate = set()

            if len(results) >= 2:

                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):

                        if D[i, j] < MIN_DISTANCE:

                            violate.add(i)
                            violate.add(j)

            for (i, (prob, bbox, centroid)) in enumerate(results):

                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                if i in violate:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            datet = str(datetime.datetime.now())
            frame = cv2.putText(frame, datet, (0, 35), font, 1,
                                (0, 255, 255), 2, cv2.LINE_AA)
            text = "Social Distancing Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

            display = 1
            if display > 0:

                image_placeholder.image(
                    frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

            if writer is not None:
                writer.write(frame)

    st.success("Design & Developed By AI4Life")

def video_analysis():
    st.title('Real Time Social Distancing Monitor System with Video')
    cuda = st.selectbox('NVIDIA CUDA GPU should be used?', ('True', 'False'))

    st.subheader('Test Demo Video Or Try Live Detection')
    all_titles = [i[0] for i in view_by_video_author()]
    option = st.selectbox('Your Name', all_titles)
    all_path = [i[0] for i in get_path_by_video_author(option)]
    option2 = st.selectbox("Select your uploaded file", all_path)


    USE_GPU = bool(cuda)
    MIN_DISTANCE = 50


    labelsPath = "yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = "yolo-coco/yolov4.weights"
    configPath = "yolo-coco/yolov4.cfg"


    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


    if USE_GPU:

        st.info("[INFO] setting preferable backend and target to CUDA...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    if st.button('Start'):

        st.info("[INFO] loading YOLO from disk...")
        st.info("[INFO] accessing video stream...")

        for i in view_by_video_author():
            if option == i[0]:
                vs = cv2.VideoCapture(option2)
            else:
                vs = cv2.VideoCapture(0)
            writer = None
            image_placeholder = st.empty()

        while True:

            (grabbed, frame) = vs.read()

            if not grabbed:
                break

            frame = imutils.resize(frame, width=700)
            results = detect_people(frame, net, ln,
                                    personIdx=LABELS.index("person"))

            violate = set()

            if len(results) >= 2:

                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):

                        if D[i, j] < MIN_DISTANCE:

                            violate.add(i)
                            violate.add(j)

            for (i, (prob, bbox, centroid)) in enumerate(results):

                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0)

                if i in violate:
                    color = (0, 0, 255)

                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            datet = str(datetime.datetime.now())
            frame = cv2.putText(frame, datet, (0, 35), font, 1,
                                (0, 255, 255), 2, cv2.LINE_AA)
            text = "Social Distancing Violations: {}".format(len(violate))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

            display = 1
            if display > 0:

                image_placeholder.image(
                    frame, caption='Live Social Distancing Monitor Running..!', channels="BGR")

            if writer is not None:
                writer.write(frame)

    st.success("Design & Developed By AI4Life")



    







if __name__ == "__main__":
    main()