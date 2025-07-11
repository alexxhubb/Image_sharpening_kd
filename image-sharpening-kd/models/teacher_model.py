import torch
from models.network_swinir import SwinIR

# Load pretrained SwinIR teacher model
def load_teacher_model(ckpt_path):
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6,6,6,6,6,6],
        embed_dim=180,
        num_heads=[6,6,6,6,6,6],
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['params'])
    model.eval()
    return model
