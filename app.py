from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
from Model.Encoder import Encoder
from Model.TransModule import TransModule, TransModule_Config
from Model.Decoder import Decoder
import torch
import torchvision.transforms as T
from torchvision.utils import save_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

def model(content_image_path, style_image_path):
    # Load Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        img_size=224,
        patch_size=2,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 2],
        nhead=[3, 6, 12],
        strip_width=[2, 4, 7],
        drop_path_rate=0.,
        patch_norm=True
    ).to(device)
    trans_config = TransModule_Config(nlayer=3, d_model=768, nhead=8, norm_first=True)
    transfer_module = TransModule(config=trans_config).to(device)
    decoder = Decoder(d_model=768, seq_input=True).to(device)

    checkpoint = torch.load('./.model/checkpoint_40000_epoch.pkl', map_location=device)  # Update the path if needed

    # Load the state dictionaries
    encoder.load_state_dict(checkpoint['encoder'])
    transfer_module.load_state_dict(checkpoint['transModule'])
    decoder.load_state_dict(checkpoint['decoder'])

    # Load the images using PIL
    content_img = Image.open(content_image_path).convert('RGB')
    style_img = Image.open(style_image_path).convert('RGB')

    # Convert the images to tensors
    content_shape = content_img.size
    size = min(content_shape[1], content_shape[0])

    # Preprocess the images
    only_tensor_transforms = T.Compose([
        T.ToTensor(),
    ])

    shape_transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])

    content_img = only_tensor_transforms(content_img).unsqueeze(0).to(device)
    style_img = shape_transform(style_img).unsqueeze(0).to(device)

    # Forward pass through encoders
    forward_content = encoder(content_img, arbitrary_input=True)  # [b, h, w, c]
    forward_style = encoder(style_img, arbitrary_input=True)      # [b, h, w, c]

    output_content, content_res = forward_content[0], forward_content[2]  # [b, c, h, w]
    output_style, style_res = forward_style[0], forward_style[2]      # [b, c, h, w]

    # Merge the features
    merged_features = transfer_module(output_content, output_style)

    # Decode the merged features
    output = decoder(merged_features, content_res)  # [b, c, h, w]

    # Convert to jpg image and save
    output = output.detach().cpu()
    save_image(output, os.path.join('results', 'result.png'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file uploads
        if 'content_image' not in request.files or 'style_image' not in request.files:
            return 'No file part'
        
        content_image = request.files['content_image']
        style_image = request.files['style_image']

        # Check if the inputs are png or jpg
        if content_image.filename == '' or style_image.filename == '':
            return 'No selected file'
        
        # Save uploaded images
        content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_image.filename))
        style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_image.filename))
        
        content_image.save(content_image_path)
        style_image.save(style_image_path)

        model(content_image_path, style_image_path)
        
        # Assuming the result image is generated and saved as 'result.png'
        result_image_path = os.path.join(app.config['RESULT_FOLDER'], 'result.png')
        
        # Display uploaded and result images after processing
        return render_template(
            'index.html', 
            content_image=os.path.basename(content_image_path),
            style_image=os.path.basename(style_image_path),
            result_image=os.path.basename(result_image_path)
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