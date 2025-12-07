# Real_time_class_room_engagement_and_interaction_monitoring_using_YOLO-V5
A real-time system that analyzes classroom engagement and student interaction using YOLOv5-based object detection and activity recognition. Includes live tracking, attention measurement, participation monitoring, and automated analytics for improving teaching effectiveness.


🚀 About This Project

This project is my attempt to answer one simple question:

“Can a classroom understand what’s happening inside it — without a teacher manually observing every student?”

Using YOLOv5, DeepSORT tracking, and a custom emotion-recognition module, this system watches a live classroom feed and tries to understand what students are doing in real time —
whether they’re paying attention, distracted, writing, using a phone, sleepy, engaged, or somewhere in between.

All the engagement cues are then blended into a single score that helps teachers instantly see how their class is doing.
Everything runs locally on edge devices (like Jetson Xavier NX), so no cloud, no privacy headache.

This project started as part of our academic research work, and evolved into a functional prototype that actually performs well in real classroom-like environments.

IEEE_conference_paper_for_major…

🎯 What This System Can Do

✔ Detect classroom activities using YOLOv5 (writing, listening, phone usage, hand-raising, sleeping, etc.)
✔ Track each student consistently across frames with DeepSORT
✔ Recognize emotions (happy, neutral, sad, angry, etc.) even with masked faces
✔ Generate real-time engagement scores (0–100)
✔ Display live analytics on a Streamlit dashboard
✔ Maintain real-time performance on edge hardware (20–25 FPS)
✔ Keep all data private — nothing leaves the device

Basically: a smart classroom assistant that doesn’t interrupt the class.

🧠 How It Works (Simple Explanation)

Camera Feed In → Video enters the system frame by frame.

YOLOv5 Detection → Students + actions are detected.

DeepSORT Tracking → Each student gets a unique ID to track over time.

Emotion Classifier → Cropped faces are analyzed for emotional cues.

Engagement Scoring → Behaviors + emotions are combined into a weighted score.

Visualization → A Streamlit dashboard shows attention levels, trends, and alerts.

All this happens continuously, in real time.

📊 Engagement Scoring (Human Explanation)

Instead of guessing who’s paying attention, the system calculates it.

Writing → +20

Fully focused → +20

Happy/Interested → +10

Neutral → 0

Sad/Bored → –20

Sleeping → –50

Phone usage → –10

These values come from our research-based design.

IEEE_conference_paper_for_major…


The final number (0–100) shows how engaged a student or the entire class is.

🖥️ Tech Stack
Core Models

YOLOv5 (custom trained)

DeepSORT for multi-object tracking

CNN-based emotion recognition

Backend & Utilities

PyTorch

OpenCV

NumPy / Pandas

FilterPy

SciPy

Streamlit (for dashboard)
