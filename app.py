from flask import Flask, render_template, request, send_from_directory
import os
import subprocess

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "crash_results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file uploaded", 400
    file = request.files['video']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Run crash_detect.py with uploaded video
    command = f'python "{os.path.join(os.getcwd(), "crash_detect.py")}" "{filepath}"'
    subprocess.run(command, shell=True)

    # Assume crash_detect.py saves 'annotated_output.mp4' and 'crash_timestamps.csv' inside RESULT_FOLDER
    output_video = os.path.join(app.config['RESULT_FOLDER'], 'annotated_output.mp4')
    csv_path = os.path.join(app.config['RESULT_FOLDER'], 'crash_timestamps.csv')

    return render_template(
        'result.html',
        video_file='annotated_output.mp4',
        csv_file='crash_timestamps.csv'
    )

@app.route('/results/<path:filename>')
def results(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

