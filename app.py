import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import tempfile

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def rgb_color_selection(image):
    # White color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)

    # Yellow color mask
    lower = np.uint8([175, 175, 0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)

    # Combine masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def canny_edge_detector(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height, width = image.shape[:2]
    triangle = np.array([
        [(100, height), (width - 100, height), (width // 2, height // 2)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def process_image(image):
    color_filtered = rgb_color_selection(image)
    canny_image = canny_edge_detector(color_filtered)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(image, lines)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo_image

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    clip = VideoFileClip(tfile.name)
    processed_clip = clip.fl_image(process_image) 
    out_path = tfile.name + "_out.mp4"
    processed_clip.write_videofile(out_path, audio=False)

    return out_path

st.title("Lane Detection App")

st.write("Upload an image or video to detect lanes.")

file = st.file_uploader("Upload here", type=["jpg", "png", "mp4"])

if file is not None:
    if file.type == "video/mp4":
        video_path = process_video(file)
        st.video(video_path)
    else:
        image = load_image(file)
        processed_image = process_image(image)
        st.image(processed_image, caption="Processed Image", use_column_width=True)
