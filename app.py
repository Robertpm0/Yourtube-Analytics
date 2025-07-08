from flask import Flask, render_template, request, redirect
import os
import subprocess
import time
TEMP_ZIP_PATH = 'shared_data/uploaded.zip'  # Streamlit will read this

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.zip'):
            # Save zip to shared_data/uploaded.zip
            os.makedirs("shared_data", exist_ok=True)
            file.save(TEMP_ZIP_PATH)
            time.sleep(10)
            # Redirect to Streamlit app
            return redirect('/analysis')

    return render_template('landing.html')

@app.route('/analysis')
def open_streamlit():
    # Launch Streamlit app if not already running
    subprocess.Popen(
        ["streamlit", "run", "main.py", "--server.headless", "true"],
        cwd=os.getcwd()
    )
    # time.sleep(20)
    return redirect('http://localhost:8501', code=302)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=False,port=10000)
