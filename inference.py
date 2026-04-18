import json
import os
from typing import Any, Dict, List

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from models import build_model

def get_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def resize_image(image: Image.Image, short_size: int = 384, max_size: int = 600) -> Image.Image:
    w, h = image.size
    if w < h:
        ow = short_size
        oh = int(short_size * h / w)
    else:
        oh = short_size
        ow = int(short_size * w / h)
        
    if max(ow, oh) > max_size:
        scale = max_size / max(ow, oh)
        ow = int(ow * scale)
        oh = int(oh * scale)
        
    return image.resize((ow, oh), Image.Resampling.BILINEAR)

def main() -> None:
    checkpoint_path = 'output/cond_detr_digit_v6/checkpoint_best.pth' # 請確認路徑
    image_dir = 'data/test'
    output_json = 'pred.json'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading final model...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = checkpoint['args']
    args.device = str(device)

    # 取得模型與內建的 postprocessors
    model, _, postprocessors = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    transform = get_transform()
    predictions: List[Dict[str, Any]] = []

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg'))]
    image_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))

    print(f"Starting inference on {len(image_files)} test images...")
    with torch.no_grad():
        for filename in tqdm(image_files):
            img_id = int(os.path.splitext(filename)[0])
            img_path = os.path.join(image_dir, filename)

            image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = image.size

            resized_image = resize_image(image)
            img_tensor = transform(resized_image).unsqueeze(0).to(device)
            
            outputs = model(img_tensor)

            # ⭐ 核心修正：直接呼叫內建 PostProcess，完美對齊 Validation 邏輯
            # 注意傳入的是 [h, w]
            orig_target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            
            # 取得單張圖片的結果
            r = results[0]
            
            # 將 [x1, y1, x2, y2] 轉為 COCO 格式的 [x, y, w, h]
            for score, label, box in zip(r['scores'], r['labels'], r['boxes']):
                x1, y1, x2, y2 = box.tolist()
                
                # 再次確保不輸出類別 0 (雖然分數極低，但保險起見直接過濾)
                cat_id = int(label.item())
                if cat_id == 0:
                    continue

                predictions.append({
                    "image_id": img_id,
                    "bbox": [max(0.0, x1), max(0.0, y1), x2 - x1, y2 - y1],
                    "score": score.item(),
                    "category_id": cat_id
                })

    # 依照 image_id 與 score 排序
    predictions.sort(key=lambda x: (x["image_id"], -x["score"]))

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to {output_json}. Your fully-aligned submission is ready!")

if __name__ == '__main__':
    main()