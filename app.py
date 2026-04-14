from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageEnhance
from Model.Encoder import Encoder
from Model.TransModule import TransModule, TransModule_Config
from Model.Decoder import Decoder
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision import models
from skimage.metrics import structural_similarity as ssim
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


def _load_image_tensor(image_path, target_shape, device):
    # target_shape should be a tuple of (H, W)
    transform = T.Compose([T.Resize(target_shape), T.ToTensor()])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)


def _compute_ssim(output_tensor, content_tensor):
    output_np = output_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    content_np = content_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    output_np = np.clip(output_np, 0.0, 1.0)
    content_np = np.clip(content_np, 0.0, 1.0)
    return float(ssim(content_np, output_np, channel_axis=2, data_range=1.0))


def _compute_lpips(tensor_a, tensor_b, device):
    try:
        import lpips
    except Exception:
        return None

    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    a_scaled = tensor_a * 2.0 - 1.0
    b_scaled = tensor_b * 2.0 - 1.0
    with torch.no_grad():
        value = lpips_model(a_scaled, b_scaled)
    return float(value.item())


def _compute_single_image_fid(output_tensor, target_tensor, device):
    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
    inception.fc = torch.nn.Identity()
    inception.eval()

    # Do not double-normalize, inception_v3(transform_input=True) expects [0, 1]
    transform = T.Compose([
        T.Resize((299, 299))
    ])

    output_in = transform(output_tensor)
    target_in = transform(target_tensor)
    with torch.no_grad():
        out_pred = inception(output_in)
        target_pred = inception(target_in)

        feat_output = out_pred.logits.squeeze(0) if hasattr(out_pred, 'logits') else out_pred.squeeze(0)
        feat_target = target_pred.logits.squeeze(0) if hasattr(target_pred, 'logits') else target_pred.squeeze(0)

    # Standard FID is the *squared* Wasserstein distance between population means.
    # For single images, the means don't average out, and the squared distance explodes.
    # Taking the square root gives the un-squared 2-Wasserstein distance (True Euclidean distance),
    # which is much more stable and visually interpretable for a one-to-one image comparison.
    return float(torch.sqrt(torch.sum((feat_output - feat_target) ** 2)).item())


def _gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


def _compute_content_and_style_losses(output_tensor, content_tensor, style_tensor, device):
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    for parameter in vgg.parameters():
        parameter.requires_grad = False

    content_layer = 21
    style_layers = [0, 5, 10, 19, 28]

    out_feat = output_tensor
    content_feat = content_tensor
    style_feat = style_tensor

    content_loss = torch.tensor(0.0, device=device)
    style_loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for idx, layer in enumerate(vgg):
            out_feat = layer(out_feat)
            content_feat = layer(content_feat)
            style_feat = layer(style_feat)

            if idx == content_layer:
                content_loss = torch.mean((out_feat - content_feat) ** 2)

            if idx in style_layers:
                gram_out = _gram_matrix(out_feat)
                gram_style = _gram_matrix(style_feat)
                style_loss = style_loss + torch.mean((gram_out - gram_style) ** 2)

    return float(content_loss.item()), float(style_loss.item())


def _compute_eval_metrics(output_tensor, content_image_path, style_image_path, device):
    h, w = output_tensor.shape[2], output_tensor.shape[3]
    target_shape = (h, w)
    content_tensor = _load_image_tensor(content_image_path, target_shape, device)
    style_tensor = _load_image_tensor(style_image_path, target_shape, device)

    metrics = {
        'SSIM': _compute_ssim(output_tensor, content_tensor),
        'LPIPS Content': _compute_lpips(output_tensor, content_tensor, device),
        'LPIPS Style': _compute_lpips(output_tensor, style_tensor, device),
        'FID': _compute_single_image_fid(output_tensor, content_tensor, device),
    }
    
    content_loss, style_loss = _compute_content_and_style_losses(
        output_tensor,
        content_tensor,
        style_tensor,
        device,
    )
    metrics['Content Loss'] = content_loss
    metrics['Style Loss'] = style_loss
    return metrics

def model(content_image_path, style_image_path, alpha=1.0):
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

    checkpoint = torch.load('./.model/checkpoint_40000_epoch.pkl', map_location=device, weights_only=False)  # Update the path if needed

    # Load the state dictionaries
    encoder.load_state_dict(checkpoint['encoder'])
    transfer_module.load_state_dict(checkpoint['transModule'])
    decoder.load_state_dict(checkpoint['decoder'])

    # Load the images using PIL
    content_img = Image.open(content_image_path).convert('RGB')
    style_img = Image.open(style_image_path).convert('RGB')

    if alpha > 0:
        # Reduce contrast
        enhancer = ImageEnhance.Contrast(style_img)
        style_img = enhancer.enhance(0.5 + 0.5 * alpha)  # 0 = no contrast (gray), 1 = original

        # Reduce saturation
        enhancer = ImageEnhance.Color(style_img)
        style_img = enhancer.enhance(0.5 + 0.5 * alpha)  # 0 = grayscale, 1 = original

    # Convert the images to tensors
    content_shape = content_img.size
    size = min(content_shape[1], content_shape[0])

    # Preprocess the images
    only_tensor_transforms = T.Compose([T.ToTensor()])
    shape_transform = T.Compose([T.Resize(size), T.ToTensor()])

    content_img = only_tensor_transforms(content_img).unsqueeze(0).to(device)
    style_img = shape_transform(style_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through encoders
        forward_content = encoder(content_img, arbitrary_input=True)  # [b, h, w, c]
        forward_style = encoder(style_img, arbitrary_input=True)      # [b, h, w, c]

        output_content, content_res = forward_content[0], forward_content[2]  # [b, c, h, w]
        output_style, style_res = forward_style[0], forward_style[2]      # [b, c, h, w]

        # Merge the features
        merged_features = transfer_module(output_content, output_style)

        # Decode the merged features
        output = decoder(merged_features, content_res)  # [b, c, h, w]

    output = output.detach()
    metrics = _compute_eval_metrics(output, content_image_path, style_image_path, device)

    # Convert to png image and save
    output_to_save = output.squeeze(0).detach().cpu()
    save_image(output_to_save, os.path.join(RESULT_FOLDER, 'result.png'))
    return metrics
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        alpha = float(request.form.get('alpha'))
        default_style = request.form.get('predefined_style')

        # Handle file uploads
        if 'content_image' not in request.files or ('style_image' not in request.files and not default_style):
            return 'No file part or style image'

        content_image = request.files['content_image']

        # Handle predefined style
        if default_style:
            style_image_path = os.path.join('static', 'predefined', f"{default_style}.jpg")
            style_image = None  # No file upload needed, using predefined style
        else:
            style_image = request.files['style_image']
            if style_image.filename == '':
                return 'No selected file for style image'
            style_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(style_image.filename))
            style_image.save(style_image_path)

        if content_image.filename == '':
            return 'No selected file for content image'

        # Save uploaded content image
        content_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(content_image.filename))
        content_image.save(content_image_path)

        eval_metrics = None

        # If alpha > 0.0, apply style transfer
        if alpha > 0.0:
            eval_metrics = model(content_image_path, style_image_path, alpha)
        else:
            # Process content image even if no style transfer is applied
            content_img = Image.open(content_image_path).convert('RGB')
            content_img.save(os.path.join(RESULT_FOLDER, 'result.png'))

        # Return the result image
        result_image_path = os.path.join(RESULT_FOLDER, 'result.png')

        # Display uploaded and result images after processing
        return render_template(
            'index.html',
            result_image=os.path.basename(result_image_path),
            eval_metrics=eval_metrics,
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
    return send_file(os.path.join(app.config['RESULT_FOLDER'], filename), mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
