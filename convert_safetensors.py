import torch
import os
import argparse
from safetensors.torch import save_file

def main(args):
    model_dir = args.model_dir
    save_dir = args.save_dir
    step = args.step

    g_model = torch.load(f"{model_dir}/G_{step}.pth", map_location="cpu")
    d_model = torch.load(f"{model_dir}/D_{step}.pth", map_location="cpu")
    dur_model = torch.load(f"{model_dir}/WD_{step}.pth", map_location="cpu")

    g_dict = {}
    for key in g_model["model"].keys():
        if key.startswith("emb_g"):
            print(key)
        else:
            g_dict[key] = g_model["model"][key]

    d_dict = {}
    for key in d_model["model"].keys():
        d_dict[key] = d_model["model"][key]

    dur_dict = {}
    for key in dur_model["model"].keys():
        dur_dict[key] = dur_model["model"][key]

    os.makedirs(save_dir, exist_ok=True)
    save_file(g_dict, os.path.join(save_dir, "G_0.safetensors"))
    save_file(d_dict, os.path.join(save_dir, "D_0.safetensors"))
    save_file(dur_dict, os.path.join(save_dir, "DW_0.safetensors"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model to pretrained")
    parser.add_argument("--model_dir", type=str, help="Directory to save the converted models")
    parser.add_argument("--save_dir", type=str, help="Directory to save the converted models")
    parser.add_argument("--step", type=int, help="Step number of the models to convert")
    args = parser.parse_args()
    main(args)