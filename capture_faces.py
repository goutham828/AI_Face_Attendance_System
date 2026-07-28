import cv2
import os

# -----------------------------
# Load Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    "haarcascades/haarcascade_frontalface_default.xml"
)

# -----------------------------
# Student ID
# -----------------------------
student_id = input("Enter Student ID: ")

# -----------------------------
# Create Dataset Folder
# -----------------------------
path = os.path.join("dataset", student_id)

os.makedirs(path, exist_ok=True)

# -----------------------------
# Open Camera
# -----------------------------
camera = cv2.VideoCapture(0)

count = 0

print("Camera Started...")
print("Press Q to Quit")

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

        count += 1

        face = gray[y:y+h, x:x+w]

        filename = os.path.join(path, f"{count}.jpg")

        cv2.imwrite(filename, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Images : {count}/50",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Capture Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if count >= 50:
        break

camera.release()

cv2.destroyAllWindows()

print("Face Capture Completed Successfully.")