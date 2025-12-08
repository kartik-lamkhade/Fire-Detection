from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import cv2
import time # Used to simulate real-time playback speed

st.set_page_config(
    page_title="Robust Video File Detection", 
    page_icon="✅"
)

st.title("🔥 Detect Fire and Smoke")
st.caption("Upload a video for robust frame-by-frame detection.")

# --- Helper Functions and Model Loading ---

@st.cache_resource 
def load_model():
    try:
        # Load your custom trained model. Ensure 'best.pt' is in the same directory.
        return YOLO("./best.pt") 
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure 'best.pt' is in the correct path.")
        return None

model = load_model()

# --- Streamlit App Layout ---

file_upload = st.file_uploader("Upload a Video File (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])

if model is None:
    st.stop() # Stop if the model didn't load

if file_upload is not None:
    # 1. Save uploaded video to a temporary file
    temp_file_path = ""
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(file_upload.read())
        temp_file_path = temp_file.name
    
    st.success("Video file uploaded successfully.")
    
    if st.button("Start Frame-by-Frame Detection"):
        st.subheader("Processing and Displaying...")
        
        # Create an empty placeholder for the video output
        video_placeholder = st.empty()
        
        # Initialize video capture using OpenCV
        cap = cv2.VideoCapture(temp_file_path)

        if not cap.isOpened():
            st.error("Error opening video file. The file may be corrupted or the codec is unsupported.")
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            st.stop()

        # Get video properties for playback speed simulation
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_time = int(1000 / fps) if fps > 0 else 30 # Time in milliseconds

        st.info(f"Video FPS: {fps:.2f}. Processing started.")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Read frame-by-frame
                ret, frame = cap.read()
                
                if not ret:
                    # End of video or error reading frame
                    break
                
                # Run YOLO prediction
                results = model.predict(frame, verbose=False)
                
                # Get the annotated frame (YOLO draws the boxes)
                annotated_frame = results[0].plot()

                # Convert BGR (OpenCV format) to RGB (Streamlit/display format)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame in the placeholder
                video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                
                frame_count += 1

                # Simulate real-time playback speed (optional, but makes it look better)
                # time.sleep(wait_time / 1000) 

            st.success("Detection complete! End of video stream.")

        except Exception as e:
            st.error(f"An error occurred during detection: {e}")
            
        finally:
            # Release the video capture object and clean up
            cap.release()
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    st.markdown("---")
    st.caption("*This approach uses native OpenCV for better video codec support.*")