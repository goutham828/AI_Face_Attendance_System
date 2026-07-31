from flask import Flask, render_template, request, send_file
import csv
import os
import subprocess
from datetime import datetime

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def home():

    total_students = 0
    total_images = 0
    today_attendance = 0

    # Count Students
    if os.path.exists("data/students.csv"):

        with open("data/students.csv", "r") as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:
                total_students += 1

    # Count Face Images
    if os.path.exists("dataset"):

        for folder in os.listdir("dataset"):

            folder_path = os.path.join("dataset", folder)

            if os.path.isdir(folder_path):
                total_images += len(os.listdir(folder_path))

    # Count Today's Attendance
    today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists("attendance.csv"):

        with open("attendance.csv", "r") as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:

                if len(row) >= 4 and row[2] == today:
                    today_attendance += 1

    return render_template(
        "index.html",
        total_students=total_students,
        total_images=total_images,
        today_attendance=today_attendance,
        today=today
    )


# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        student_id = request.form["student_id"]
        student_name = request.form["student_name"]

        os.makedirs("data", exist_ok=True)

        file_path = "data/students.csv"

        if not os.path.exists(file_path):

            with open(file_path, "w", newline="") as file:

                writer = csv.writer(file)
                writer.writerow(["Student_ID", "Name"])

        duplicate = False

        with open(file_path, "r") as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:

                if row and row[0] == student_id:
                    duplicate = True
                    break

        if duplicate:

            return """
            <script>
            alert("Student ID already exists!");
            window.location.href="/register";
            </script>
            """

        with open(file_path, "a", newline="") as file:

            writer = csv.writer(file)
            writer.writerow([student_id, student_name])

        return """
        <script>
        alert("Student Registered Successfully");
        window.location.href="/";
        </script>
        """

    return render_template("register.html")


# ---------------- VIEW STUDENTS ----------------
@app.route("/students")
def students():

    student_list = []

    if os.path.exists("data/students.csv"):

        with open("data/students.csv", "r") as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:
                student_list.append(row)

    return render_template(
        "students.html",
        students=student_list
    )


# ---------------- CAPTURE FACE ----------------
@app.route("/capture")
def capture():

    subprocess.Popen(
        ["python", "capture_faces.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    return """
    <script>
    alert("Face Capture Started");
    window.location.href="/";
    </script>
    """


# ---------------- TRAIN MODEL ----------------
@app.route("/train")
def train():

    subprocess.Popen(
        ["python", "train_model.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    return """
    <script>
    alert("Model Training Started");
    window.location.href="/";
    </script>
    """


# ---------------- START ATTENDANCE ----------------
@app.route("/start")
def start():

    subprocess.Popen(
        ["python", "recognize.py"],
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

    return """
    <script>
    alert("Attendance Started");
    window.location.href="/";
    </script>
    """


# ---------------- ATTENDANCE ----------------
@app.route("/attendance")
def attendance():

    records = []

    search = request.args.get("search", "").lower()

    if os.path.exists("attendance.csv"):

        with open("attendance.csv", "r") as file:

            reader = csv.reader(file)

            next(reader, None)

            for row in reader:

                if len(row) < 4:
                    continue

                if search == "":
                    records.append(row)

                else:

                    if search in " ".join(row).lower():
                        records.append(row)

    return render_template(
        "attendance.html",
        records=records,
        search=search
    )


# ---------------- DOWNLOAD CSV ----------------
@app.route("/download")
def download():

    if os.path.exists("attendance.csv"):

        return send_file(
            "attendance.csv",
            as_attachment=True
        )

    return "Attendance file not found."


# ---------------- RUN ----------------
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)