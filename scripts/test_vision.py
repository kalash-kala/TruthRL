
import os
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd

def test_model_vision():
    model_path = "/home/debarpanb1/models/Qwen2.5-VL-3B-Instruct"
    data_path = "/home/debarpanb1/kalashkala/visual-spatial-reasoning/truthrl-sample/parquet/test.parquet"
    
    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    print(f"Loading model from {model_path}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
        trust_remote_code=True
    )
    model.eval()

    print(f"Loading sample from {data_path}...")
    df = pd.read_parquet(data_path)
    row = df.iloc[0]
    
    # Extract image path
    image_path = "unknown"
    if 'images' in row and len(row['images']) > 0:
        first_img = row['images'][0]
        if isinstance(first_img, dict):
            image_path = first_img['image']
        else:
            image_path = first_img
            
    print(f"Testing with image path: {image_path}")

    # Prepare message exactly like the eval script
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Describe this image in detail. What objects do you see and where are they?"}
            ]
        }
    ]

    # Preprocess
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    print(f"image_inputs length: {len(image_inputs) if image_inputs else 0}")
    if image_inputs:
        print(f"First image input type: {type(image_inputs[0])}")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    # Generate
    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "="*50)
    print("MODEL RESPONSE:")
    print("="*50)
    print(output_text)
    print("="*50)

if __name__ == "__main__":
    test_model_vision()
