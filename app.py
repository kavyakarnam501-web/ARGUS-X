import os
import tempfile
import cv2
import streamlit as st
import numpy as np

from PIL import Image
from detector import PersonDetector
from tracker import IdentityTracker
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av


st.set_page_config(
    page_title="ARGUS-X",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 ARGUS-X")
st.subheader("AI-Powered Object Detection and Tracking System")

st.write(
    "Detect people, vehicles and other objects using "
    "image, video and live camera input."
)


@st.cache_resource
def load_detector():
    return PersonDetector()


def draw_results(frame, objects):

    result = frame.copy()

    for obj in objects:

        x1, y1, x2, y2 = obj["bbox"]

        label = (
            f'{obj["identity_label"]} | '
            f'{obj["confidence"]:.2f} | '
            f'{obj["movement"]}'
        )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            result,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return result


def show_statistics(objects):

    people = sum(
        obj["class_id"] == 0 for obj in objects
    )

    total = len(objects)

    moving = sum(
        obj["movement"] == "Moving" for obj in objects
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("People detected", people)
    c2.metric("Total objects", total)
    c3.metric("Moving objects", moving)


detector = load_detector()

mode = st.sidebar.radio(
    "Select Input Mode",
    [
        "Image Detection",
        "Video Detection",
        "Live Webcam"
    ]
)

st.sidebar.success("YOLOv8 Model Loaded")


# ---------------------------------
# IMAGE DETECTION
# ---------------------------------

if mode == "Image Detection":

    st.header("🖼️ Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")
        frame = cv2.cvtColor(
            np.array(image), cv2.COLOR_RGB2BGR
        )

        if st.button("Detect Objects"):

            with st.spinner("Detecting objects..."):

                detections = detector.detect(
                    frame, track=False
                )

                # Use a fresh tracker for this image.
                image_tracker = IdentityTracker()

                objects = image_tracker.update(
                    detections, 1
                )

                result = draw_results(frame, objects)

                show_statistics(objects)

                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    caption="ARGUS-X Detection Result",
                    use_container_width=True
                )

                st.download_button(
                    "Download Detected Image",
                    data=cv2.imencode(
                        ".jpg", result
                    )[1].tobytes(),
                    file_name="argus_detection.jpg",
                    mime="image/jpeg"
                )


# ---------------------------------
# VIDEO DETECTION
# ---------------------------------

elif mode == "Video Detection":

    st.header("🎬 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("Start Video Detection"):

            temp_input = None
            temp_output = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as input_file:

                    input_file.write(uploaded_video.getvalue())
                    temp_input = input_file.name

                cap = cv2.VideoCapture(temp_input)

                if not cap.isOpened():
                    st.error("Cannot open uploaded video.")
                else:

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 25

                    width = int(
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    )
                    height = int(
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp4"
                    ) as output_file:
                        temp_output = output_file.name

                    writer = cv2.VideoWriter(
                        temp_output,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height)
                    )

                    video_tracker = IdentityTracker()

                    frame_count = 0
                    total_frames = int(
                        cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    )

                    progress = st.progress(0)
                    status = st.empty()

                    while True:

                        ret, frame = cap.read()

                        if not ret:
                            break

                        frame_count += 1

                        detections = detector.detect(
                            frame, track=True
                        )

                        objects = video_tracker.update(
                            detections, frame_count
                        )

                        result = draw_results(
                            frame, objects
                        )

                        writer.write(result)

                        if total_frames > 0:
                            progress.progress(
                                min(
                                    frame_count / total_frames,
                                    1.0
                                )
                            )

                        status.write(
                            f"Processing frame {frame_count}"
                        )

                    cap.release()
                    writer.release()

                    with open(temp_output, "rb") as output_file:

                        video_bytes = output_file.read()

                    st.success("Video processing completed!")

                    st.video(video_bytes)

                    st.download_button(
                        "Download Processed Video",
                        data=video_bytes,
                        file_name="argus_tracked_video.mp4",
                        mime="video/mp4"
                    )

            except Exception as error:
                st.error(f"Video processing error: {error}")

            finally:
                if temp_input and os.path.exists(temp_input):
                    os.remove(temp_input)

                if temp_output and os.path.exists(temp_output):
                    os.remove(temp_output)


# ---------------------------------
# LIVE WEBCAM
# ---------------------------------

elif mode == "Live Webcam":

    st.header("📷 Live Webcam Detection")

    st.info(
        "Allow camera access in your browser when prompted. "
        "The webcam runs on this computer."
    )

    class LiveProcessor(VideoProcessorBase):

        def __init__(self):
            self.detector = PersonDetector()
            self.tracker = IdentityTracker()
            self.frame_count = 0

        def recv(self, video_frame):

            frame = video_frame.to_ndarray(
                format="bgr24"
            )

            self.frame_count += 1

            detections = self.detector.detect(
                frame, track=True
            )

            objects = self.tracker.update(
                detections, self.frame_count
            )

            result = draw_results(frame, objects)

            return av.VideoFrame.from_ndarray(
                result, format="bgr24"
            )

    webrtc_streamer(
        key="argus-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

st.divider()

st.caption(
    "ARGUS-X | YOLOv8 | OpenCV | ByteTrack | Streamlit"
)