from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file uploads
        if 'content_image' not in request.files or 'style_image' not in request.files:
            return 'No file part'
        
        content_image = request.files['content_image']
        style_image = request.files['style_image']
        
        if content_image.filename == '' or style_image.filename == '':
            return 'No selected file'
        
        # Save uploaded images
        content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_image.filename))
        style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_image.filename))
        
        content_image.save(content_image_path)
        style_image.save(style_image_path)
        
        # Assuming the result image is generated and saved as 'result.jpg'
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result.jpg')
        
        # Display uploaded and result images after processing
        return render_template(
            'index.html', 
            content_image=os.path.basename(content_image_path),
            style_image=os.path.basename(style_image_path),
            result_image='result.jpg'
        )

    # If GET request, show the form without the result image
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/jpeg')

# Route to serve the result image
@app.route('/results/<filename>')
def result_file(filename):
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
