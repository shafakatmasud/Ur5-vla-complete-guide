#!/usr/bin/env python3
"""Read-only OpenVLA adapter loader. Never connects to or commands a robot."""
import argparse
import json
from pathlib import Path

from train import MODEL, MODEL_REVISION, verify_checkpoint
from dataset import DATASET_KEY


def load_checkpoint(checkpoint, device='cuda:0'):
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    checkpoint = Path(checkpoint)
    verify_checkpoint(checkpoint)
    run = json.loads((checkpoint / 'run_config.json').read_text())
    signature = run['signature']
    if signature['model'] != MODEL or signature['model_revision'] != MODEL_REVISION:
        raise ValueError('Checkpoint base model/revision differs from this loader')
    processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REVISION, trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(MODEL, revision=MODEL_REVISION,
             trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
             attn_implementation='eager').to(device)
    statistics = json.loads((checkpoint / 'dataset_statistics.json').read_text())
    base.norm_stats = statistics
    base.config.norm_stats = statistics
    model = PeftModel.from_pretrained(base, checkpoint).eval()
    return model, processor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--image', type=Path, help='Optional base-camera RGB image for one read-only prediction')
    parser.add_argument('--task', default='Pick up the orange cube and place it in the bowl')
    args = parser.parse_args()
    marker = verify_checkpoint(args.checkpoint)
    print('Verified adapter checkpoint, optimizer step:', marker['step'])
    print('Action contract: absolute UR5 joint radians (6), gripper 0=closed / 1=open (1).')
    if args.image is None:
        print('Inspection only: no model loaded; no command publishers.')
        return
    import torch
    from PIL import Image
    model, processor = load_checkpoint(args.checkpoint)
    prompt = f'In: What action should the robot take to {args.task.lower()}?\nOut:'
    with Image.open(args.image) as image:
        batch = processor(prompt, image.convert('RGB'), return_tensors='pt')
    batch = {k: v.to('cuda:0', dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda:0')
             for k, v in batch.items()}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        action = model.predict_action(**batch, unnorm_key=DATASET_KEY, do_sample=False)
    print('READ-ONLY PREDICTION, never sent to robot:', action.tolist())
    print('Thresholded gripper intent:', 'OPEN' if action[6] >= .5 else 'CLOSE')


if __name__ == '__main__':
    main()
