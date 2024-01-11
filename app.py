import streamlit as st
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO
import cvzone
import time
import torch
import os
from PIL import Image

demo_image = "crowd-concert-monterrey-city-mexico-concert-called-pal-norte-croud-attending-to-concert-monterrey-103502412.jpg"
demo_video = "Metro.mp4"


# setting page configuration
st.set_page_config(layout='centered', initial_sidebar_state='auto')

with open('style.css') as s:
    st.markdown(f'<style>{s.read()}</style>', unsafe_allow_html=True)

print(f'{torch.backends.mps.is_available()}')

# Sidebar Headers
st.sidebar.image("B_metro.png", width=70)
st.sidebar.subheader("Model Weight and Media Format Selection")
st.sidebar.markdown("""---""")

# sidebar Media format ( APP MODE )
st.sidebar.subheader("Pages:")

# st.sidebar.markdown("Choose the Media Format Mode:")
app_mode = st.sidebar.radio('', ['Home', 'Image', 'Video', 'Camera'])
st.sidebar.markdown("""---""")

# sidebar Model Weights
model_weights_dir = "../Yolo weights"
model_weights_files = os.listdir(model_weights_dir)
model_weights_files = model_weights_files[0:2] + model_weights_files[3:6]
st.sidebar.subheader("Select Model Weight:")
selected_weight = st.sidebar.selectbox("", model_weights_files[:])
# print(model_weights_files)
if selected_weight:
    model_weight_path = os.path.join(model_weights_dir, selected_weight)
    st.sidebar.text(f"Selected Model Weight: {selected_weight}")
st.sidebar.markdown("""---""")

# metrics
col1, col2, col3 = st.columns(3)

# Home Mode
if app_mode == "Home":
    st.header('METRO COACH OCCUPANCY STATUS', divider='rainbow')

    st.subheader(":orange[Scope:]")
    st.markdown('''The scope of a project on a metro coach occupancy predictor would typically involve collecting data
on metro coach ridership and occupancy levels. This would include identifying the number of
passengers that board and alight from each coach, the time of day, the day of the week, and any
special events that may affect ridership. Once this data is collected, the project would involve
analyzing it to identify patterns and trends. This would involve using statistical and machine learning
techniques to identify factors that influence occupancy levels. The next step would be to develop a
model that can predict coach occupancy levels based on the identified factors. The model would
need to be accurate and reliable to ensure that it can be used to make informed decisions about
scheduling and capacity planning. Finally, the project would involve testing and validating the
model to ensure that it can be effectively used to predict coach occupancy levels in real-time. The
overall goal of the project would be to develop a tool that can help improve the efficiency and
safety of the metro system by anticipating and managing coach occupancy levels.''')

    st.subheader(":orange[Aim of metro coach occupancy project :]")
    st.markdown("""
                    The aim of a metro coach occupancy predictor project is to develop a tool that can accurately
                predict the occupancy levels of metro coaches in real-time. By collecting data on ridership patterns
                and using statistical and machine learning techniques to analyze it, the project aims to identify the
                factors that influence coach occupancy levels. The ultimate goal is to develop a model that can
                accurately predict coach occupancy based on these factors, enabling metro authorities to manage
                capacity more efficiently and effectively. The project aims to improve the safety and comfort of
                metro passengers by ensuring that coaches are not overcrowded and that ridership is balanced
                across the system. By anticipating occupancy levels, the predictor can also help reduce wait times
                for passengers and improve the overall efficiency of the metro system. Ultimately, the aim of the
                project is to develop a tool that can help metro authorities make informed decisions about
                scheduling and capacity planning, improving the overall performance of the metro system and
                enhancing the passenger experience.
                """, unsafe_allow_html=True)

    st.subheader(":green[1. Optimizing routes and schedules:] ")
    st.markdown(
        "By understanding how many passengers are using each route and at what times, the metro system can optimize routes and schedules to better meet demand and reduce wait times for passengers.")

    st.subheader(":green[2. Reducing overcrowding:]")
    st.markdown(
        "By tracking passenger occupancy in real-time, the metro system can take steps to prevent overcrowding and improve passenger safety, such as adding additional coaches or adjusting schedules to distribute passenger demand more evenly.")

    st.subheader(":green[3. Improving fare pricing and revenue management: ]")
    st.markdown(
        "By understanding passenger demand, the metro system can adjust fares to better match demand and optimize revenue.")

    st.subheader(":green[4. Improving passenger safety:] ")
    st.markdown(
        "By monitoring passenger occupancy, the metro system can identify when coaches are overcrowded and take steps to address the issue, improving passenger safety.")

    st.markdown('---')
    st.subheader(":orange[Summary:]")
    st.markdown(
        "Overall, the aim of a metro coach occupancy project is to improve the experience of using the metro system for passengers and to optimize the use of resources by the metro system. This could conclude improvements in route optimization, overcrowding reduction, fare pricing, or passenger safety that were achieved as a result of the project.")

# Model
path = f"../Yolo weights/{selected_weight}"
model = YOLO(path)

# Image Mode
if app_mode == 'Image':
    st.header("Image mode:", divider='rainbow')
    # st.header("", divider="rainbow")

    a, b, c = st.columns(3)

    with a:
        a = st.empty()
        st.markdown(':blue[Count]')
    with b:
        b = st.empty()
        st.markdown(':red[Space occupied]')
    with c:
        c = st.empty()
        st.markdown(":green[Coach]")

    st.text("Upload image")
    img_uploaded = st.file_uploader("", type=['jpeg', 'jpg', 'png'])

    if img_uploaded is not None:
        image = np.array(Image.open(img_uploaded))
        st.image(image, caption="Uploaded Image", width=100)
    else:
        image = np.array(Image.open(demo_image))
        st.image(image, caption="Demo Image", width=100)

    path = f"../Yolo weights/{selected_weight}"
    st.text(path)

    model = YOLO(path)

    start = st.button("Start")
    if start:
        res = model.predict(image, device="mps")
        info = res[0]
        count = len(info.boxes)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted, width=600)
        a.subheader(f":blue[{count}]")
        b.subheader(f":red[{(count / 200) * 100}%]")
        c.subheader(f":green[]")

# Video Mode
if app_mode == 'Video':
    st.header("Video Mode", divider='rainbow')
    st.sidebar.markdown("""---""")
    path = f"../Yolo weights/{selected_weight}"
    model = YOLO(path)

    a, b, c = st.columns(3)

    with a:
        a = st.empty()
        st.markdown(':blue[Count]')
    with b:
        b = st.empty()
        st.markdown(':red[Space occupied]')
    with c:
        c = st.empty()
        st.markdown(":green[Coach]")

    # video_mode = st.radio("File Upload or Webcam", ['File Upload', 'Webcam'])
    f = st.file_uploader('Upload Video', type=['mp4', 'mov', 'avi'])
    tfile = tempfile.NamedTemporaryFile(delete=False)

    if f:
        tfile.write(f.read())
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
    else:
        vf = cv2.VideoCapture(demo_video)
        stframe = st.empty()

    start = st.button("Start Detection")
    if start:
        while vf.isOpened():
            ret, frame = vf.read()

            if ret:
                res = model.predict(frame, device='mps')
                info = res[0]
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                stframe.image(res_plotted,
                              caption='Detected Video',
                              channels="BGR",
                              use_column_width=True
                              )
                count = len(info.boxes)
                a.subheader(f":blue[{count}]")
                b.subheader(f":red[{(count / 200) * 100}%]")
                c.subheader(f":green[]")

            else:
                st.text("cant receive frames:")
                vf.release()
                break


# Camera Mode
if app_mode == 'Camera':
    st.header("Camera Mode", divider='rainbow')
    st.sidebar.markdown("""---""")
    a, b, c = st.columns(3)
    selected = st.selectbox("Select between Webcam mode and External Camera mode:",
                            ['Webcam', 'External Camera 1', 'External Camera 2'])

    stframe = st.empty()

    with a:
        a = st.empty()
        st.markdown(':blue[Count]')
    with b:
        b = st.empty()
        st.markdown(':red[Space occupied]')
    with c:
        c = st.empty()
        st.markdown(":green[Coach]")

    if selected == "Webcam":
        btn = st.button("Open Cam")
        if btn:
            vf = cv2.VideoCapture(0)
            kill = st.button("Stop")

            while vf.isOpened():
                ret, frame = vf.read()
                if kill:
                    exit()

                if ret:
                    res = model.predict(frame, device='mps')
                    info = res[0]
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    stframe.image(res_plotted,
                                  caption='Detected Video',
                                  channels="BGR",
                                  use_column_width=True
                                  )
                    count = len(info.boxes)
                    a.subheader(f":blue[{count}]")
                    b.subheader(f":red[{(count / 200) * 100}%]")
                    c.subheader(f":green[]")
                else:
                    st.text("cant receive frames:")
                    vf.release()
                    break

    elif selected == 'External Camera 1':
        btn = st.button("Open Cam")
        if btn:
            vf = cv2.VideoCapture(1)
            kill = st.button("Stop")

            while vf.isOpened():
                ret, frame = vf.read()
                if kill:
                    exit()

                if ret:
                    res = model.predict(frame, device='mps')
                    info = res[0]
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    stframe.image(res_plotted,
                                  caption='Detected Video',
                                  channels="BGR",
                                  use_column_width=True
                                  )
                    count = len(info.boxes)
                    a.subheader(f":blue[{count}]")
                    b.subheader(f":red[{(count / 200) * 100}%]")
                    c.subheader(f":green[]")
                else:
                    st.text("cant receive frames:")
                    vf.release()
                    break


    elif selected == 'External Camera 2':
        btn = st.button("Open Cam")
        if btn:
            vf = cv2.VideoCapture(2)
            kill = st.button("Stop")

            while vf.isOpened():
                ret, frame = vf.read()
                if kill:
                    exit()

                if ret:
                    res = model.predict(frame, device='mps')
                    info = res[0]
                    result_tensor = res[0].boxes
                    res_plotted = res[0].plot()
                    stframe.image(res_plotted,
                                  caption='Detected Video',
                                  channels="BGR",
                                  use_column_width=True
                                  )
                    count = len(info.boxes)
                    a.subheader(f":blue[{count}]")
                    b.subheader(f":red[{(count / 200) * 100}%]")
                    c.subheader(f":green[]")

                else:
                    st.text("cant receive frames:")
                    vf.release()
                    break


# Bottom Credits
st.header("", divider="rainbow")
st.subheader(":blue[Credits]")

cred1, cred2, cred3, cred4 = st.columns(4)

with cred1:
    st.markdown('''
    Abhinav Chandra
    ''''')
    st.text("200303124557")
with cred2:
    st.text("Shruti Roy")
    st.text("200303124577")

with cred3:
    st.text("Keshav Kant Gupta ")
    st.text("200303124565")

with cred4:
    st.text("Himanshu Pandey")
    st.text("200303124587")
st.markdown("[Link Text](http://localhost:8502)")