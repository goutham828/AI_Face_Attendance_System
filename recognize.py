import cv2
import csv
import os
from datetime import datetime

# -----------------------------
# Load Student Details
# -----------------------------
students = {}

if os.path.exists("data/students.csv"):

    with open("data/students.csv", "r") as file:

        reader = csv.DictReader(file)

        for row in reader:
            students[int(row["Student_ID"])] = row["Name"]

print("Students Loaded:", students)

# -----------------------------
# Load Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    "haarcascades/haarcascade_frontalface_default.xml"
)

# -----------------------------
# Load Trained Model
# -----------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainer/trainer.yml")

# -----------------------------
# Attendance File
# -----------------------------
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):

    with open(attendance_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Student_ID",
            "Student_Name",
            "Date",
            "Time"
        ])

# -----------------------------
# Already Marked Today
# -----------------------------
marked = set()

today = datetime.now().strftime("%Y-%m-%d")

with open(attendance_file, "r") as file:

    reader = csv.reader(file)

    next(reader, None)

    for row in reader:

        if len(row) >= 4:

            try:
                if row[2] == today:
                    marked.add(int(row[0]))
            except:
                pass

# -----------------------------
# Open Camera
# -----------------------------
camera = cv2.VideoCapture(0)

print("--------------------------------")
print("Camera Started")
print("Press Q to Exit")
print("--------------------------------")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        print(f"ID : {student_id}   Confidence : {confidence:.2f}")

        if confidence < 70:

            student_name = students.get(student_id, "Unknown")

            if student_id not in marked:

                now = datetime.now()

                with open(attendance_file, "a", newline="") as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        student_id,
                        student_name,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    ])

                marked.add(student_id)

                message = "Attendance Marked"
                color = (0, 255, 0)

                print(f"Attendance Marked : {student_name}")

            else:

                message = "Already Marked Today"
                color = (0, 255, 255)

                print(f"Attendance Already Marked : {student_name}")

            cv2.putText(
                frame,
                student_name,
                (x, y - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                frame,
                message,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        else:

            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

    cv2.imshow("AI Face Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("--------------------------------")
print("Face Recognition Stopped")
print("--------------------------------")

input("\nPress Enter to close...")