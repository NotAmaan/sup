
import os
import runpod
from runpod.serverless.utils import rp_upload
import asyncio
import subprocess
import torch.cuda
from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from PIL import Image
import time
import requests
from io import BytesIO

# Use the MODEL environment variable; fallback to a default if not set
mode_to_run = os.getenv("MODE_TO_RUN", "pod")

print("------- ENVIRONMENT VARIABLES -------")
print("Mode running: ", mode_to_run)
print("------- -------------------- -------")

# Global variable to hold the model
SUPIR_MODEL = None

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

# =====================================================================
def setup_model(args, device):
    # load the config
    if args.sampler_mode == "TiledRestoreEDMSampler":
        config = "options/SUPIR_v0_tiled.yaml"
    else:
        config = "options/SUPIR_v0.yaml"

    # create SUPIR model
    model = create_SUPIR_model(config, SUPIR_sign=args.SUPIR_sign)

    # precision settings
    if args.loading_half_params:
        model = model.half()
    if args.use_tile_vae:
        model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
    
    model.ae_dtype = convert_dtype(args.ae_dtype)
    model.model.dtype = convert_dtype(args.diff_dtype)

    # move the model to device (cuda or cpu)
    model = model.to(device)

    # if using TiledRestoreEDMSampler
    if args.sampler_mode == "TiledRestoreEDMSampler":
        # set/override tile size and tile stride
        model.sampler.tile_size = args.sampler_tile_size
        model.sampler.tile_stride = args.sampler_tile_stride
    
    return model

# =====================================================================
def process_image(model, args, device):
    img_path = args.img_path
    img_name = os.path.splitext(os.path.basename(img_path))[0]  # Get base filename without extension
    
    try:
        if img_path.startswith("http"): # if it's a URL
            response = requests.get(img_path)
            response.raise_for_status() # Raise an exception for bad status codes
            LQ_ips = Image.open(BytesIO(response.content))
        else: # if it's a local path
            LQ_ips = Image.open(img_path)
    except FileNotFoundError:
        return None, f"Error: Input image not found at {img_path}"
    except requests.exceptions.RequestException as e:
        return None, f"Error downloading image from {img_path}: {e}"
    except Exception as e:
        return None, f"Error opening image {img_path}: {e}"
    
    # update the input image
    LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upscale=args.upscale, min_size=args.min_size)
    LQ_img = LQ_img.unsqueeze(0).to(device)[:, :3, :, :]
    
    # Image caption(s)
    # captions = [args.img_caption]
    
    # Diffusion Process
    # batchify_sample() is in SUPIR/models/SUPIR_model.py
    samples = model.batchify_sample(LQ_img, args.img_caption, 
                                    num_steps=args.edm_steps, 
                                    restoration_scale=args.restoration_scale, 
                                    s_churn=args.s_churn,
                                    s_noise=args.s_noise,
                                    cfg_scale_start=args.cfg_scale_start,                                     
                                    cfg_scale_end=args.cfg_scale_end, 
                                    control_scale_start=args.control_scale_start,
                                    control_scale_end=args.control_scale_end, 
                                    seed=args.seed,
                                    num_samples=args.num_samples, 
                                    p_p=args.a_prompt, 
                                    n_p=args.n_prompt, 
                                    color_fix_type=args.color_fix_type,
                                    skip_denoise_stage=args.skip_denoise_stage)
    
    return samples, h0, w0, img_name

# =====================================================================
def save_results(samples, h0, w0, img_name, save_dir):
    output_base_name = f"{img_name}_SUPIR"  # Construct a base name for output
    saved_paths = []
    
    for _i, sample in enumerate(samples):
        # Determine initial filename
        base_filename = f'{output_base_name}_{_i}' if len(samples) > 1 else output_base_name
        extension = '.png'
        output_filename = f'{base_filename}{extension}'
        save_path = os.path.join(save_dir, output_filename)
        
        # Check if file exists and append index if necessary
        counter = 1
        while os.path.exists(save_path):
            output_filename = f'{base_filename}_{counter}{extension}'
            save_path = os.path.join(save_dir, output_filename)
            counter += 1
        
        # Save the image
        Tensor2PIL(sample, h0, w0).save(save_path)
        saved_paths.append(save_path)
        print(f"Saved output image to: {save_path}")
    
    return saved_paths

def run_supir(event):
    global SUPIR_MODEL
    input_params = event.get("input", {})
    
    # Set default values for arguments
    args = Args(
        img_path=input_params.get("img_path"),
        save_dir=input_params.get("save_dir", "output"),
        upscale=input_params.get("upscale", 2),
        SUPIR_sign=input_params.get("SUPIR_sign", "Q"),
        sampler_mode=input_params.get("sampler_mode", "TiledRestoreEDMSampler"),
        seed=input_params.get("seed", 1234567891),
        min_size=input_params.get("min_size", 1024),
        edm_steps=input_params.get("edm_steps", 50),
        restoration_scale=input_params.get("restoration_scale", -1),
        s_churn=input_params.get("s_churn", 5),
        s_noise=input_params.get("s_noise", 1.003),
        num_samples=input_params.get("num_samples", 1),
        img_caption=input_params.get("img_caption", ""),
        a_prompt=input_params.get("a_prompt", 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations.'),
        n_prompt=input_params.get("n_prompt", 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth'),
        color_fix_type=input_params.get("color_fix_type", "Wavelet"),
        cfg_scale_start=input_params.get("cfg_scale_start", 2.0),
        cfg_scale_end=input_params.get("cfg_scale_end", 4.0),
        control_scale_start=input_params.get("control_scale_start", 0.9),
        control_scale_end=input_params.get("control_scale_end", 0.9),
        loading_half_params=input_params.get("loading_half_params", True),
        ae_dtype=input_params.get("ae_dtype", "bf16"),
        diff_dtype=input_params.get("diff_dtype", "fp16"),
        use_tile_vae=input_params.get("use_tile_vae", False),
        encoder_tile_size=input_params.get("encoder_tile_size", 512),
        decoder_tile_size=input_params.get("decoder_tile_size", 64),
        skip_denoise_stage=input_params.get("skip_denoise_stage", False),
        sampler_tile_size=input_params.get("sampler_tile_size", 128),
        sampler_tile_stride=input_params.get("sampler_tile_stride", 64)
    )

    if not args.img_path:
        return {"error": "img_path is required"}

    # Check for CUDA availability
    if torch.cuda.device_count() >= 1:
        device = 'cuda:0'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA not available. Falling back to CPU. Warning: This will be significantly slower.")

    # Create output directory if not exist
    os.makedirs(args.save_dir, exist_ok=True)
        
    # Setup SUPIR model if it's not already loaded
    if SUPIR_MODEL is None:
        SUPIR_MODEL = setup_model(args, device)
    
    # Process the image
    result, error_message = process_image(SUPIR_MODEL, args, device)
    if error_message:
        return {"error": error_message}

    if result is not None:
        samples, h0, w0, img_name = result
        
        # Save results
        saved_paths = save_results(samples, h0, w0, img_name, args.save_dir)
        
        # Upload the first image to RunPod bucket
        if saved_paths:
            # Check if running on RunPod
            if os.getenv("RUNPOD_API_KEY"):
                image_url = rp_upload.upload_image_to_runpod_bucket(saved_paths[0])
                return {"image_url": image_url}
            else:
                return {"image_path": saved_paths[0]}

    return {"error": "Failed to process image"}


async def handler(event):
    return run_supir(event)

if mode_to_run == "pod":
    # Start the Gradio UI
    print("Starting Gradio UI...")
    subprocess.run(["python3", "run_supir_gradio.py", "--listen"])
    
else: 
    runpod.serverless.start({
        "handler": handler,
        "concurrency_modifier": lambda current: 1,
    })
