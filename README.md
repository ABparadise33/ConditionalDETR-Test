# Conditional DETR for Digit Detection (VR 2026 Spring HW2)

本專案將 **Conditional DETR** 應用於 2026 春季課程「Visual Recognition using Deep Learning」的 Homework 2：Digit Detection 任務。

## 任務簡介
- **目標**：偵測影像中的數字。
- **評分標準**：
    - CodaBench 競賽表現 (80%)
    - 程式碼品質與可靠性 (5%)
    - 報告與方法說明 (15%)

## 方法說明 (Methodology)
本實驗採用 **Conditional DETR** 作為基準模型。相較於原始 DETR，Conditional DETR 透過解耦內容與空間查詢 (Content and Spatial Queries)，能更快收斂並獲得更精確的邊界框預測，適合用於數字偵測這種對於空間位置敏感的任務。

## 環境安裝
建議使用 `conda` 或 `virtualenv` 建立環境：

```bash
# 安裝必要套件
pip install -r requirements.txt
```

*主要依賴項包含：PyTorch, torchvision, pycocotools, scipy 等。*

## 資料集準備
請依照以下目錄結構存放資料（此路徑已加入 `.gitignore`，不會上傳至 GitHub）：

```text
data/
  ├── train/         # 訓練集圖片
  ├── val/           # 驗證集圖片
  └── annotations/   # COCO 格式的標註檔
```

## 訓練模型 (Training)
使用以下指令啟動訓練（以 ResNet-50 為骨幹網路）：

```bash
python main.py \
    --dataset_file coco \
    --coco_path ./data \
    --output_dir ./output \
    --resume [PRETRAINED_WEIGHT_PATH]
```

## 推論與提交 (Inference)
執行 `inference.py` 產生預測結果，以利提交至 CodaBench 平台：

```bash
python inference.py \
    --resume ./output/checkpoint.pth \
    --image_path ./data/test \
    --output_json predictions.json
```

## 目錄架構
- `models/`: 包含 Conditional DETR 的模型定義。
- `datasets/`: 包含 COCO 格式資料載入器。
- `util/`: 包含 Bounding Box 運算與分散式訓練工具。
- `output/`: 訓練過程中的權重檔（已設定排除上傳 `.pth`）。
- `inference.py`: 用於產生測試集預測結果。

## 程式碼規範
本專案遵循：
- **PEP8** 規範。
- **Google Python Style Guide**。
- 提供乾淨的訓練與推論指令說明。

## 引用
```bibtex
@inproceedings{meng2021conditional,
  title={Conditional DETR for Fast Training Convergence},
  author={Meng, Depu and Chen, Xiaokang and Zhang, Zejia and Hou, Liang and Wu, Jianfei and Wang, Xuejin and He, Nanyue and Zhang, Bin and Shen, Jianfeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3651--3660},
  year={2021}
}
```

---

### 如何編輯與上傳：
1. 在 Vast.ai 的 Terminal 輸入：`nano README.md`
2. 將上面的內容貼上，按 `Ctrl+O` 存檔，`Ctrl+X` 離開。
3. 執行 Git 上傳步驟：
   ```bash
   git add README.md
   git commit -m "docs: add README for HW2 digit detection"
   git push origin main
   ```
