# AutoMaterial: 材料性質預測案例

## 任務概述

本案例專注於**晶體結構形成能預測**（Formation Energy Prediction），使用 MatBench 資料集中的 `matbench_mp_e_form` 任務。

### 任務描述
- **輸入**：晶體結構（CIF 格式）
- **輸出**：形成能（Formation Energy，單位：eV/atom）
- **評估指標**：MAE (Mean Absolute Error)
- **資料集**：MatBench matbench_mp_e_form（132,752 個樣本）

## 環境設置

### 必要依賴

```bash
# PyTorch 和 PyTorch Geometric
pip install torch torch-geometric

# 材料科學工具
pip install pymatgen

# MatBench 資料集
pip install matbench

# 其他依賴
pip install numpy tqdm
```

### 完整安裝指令

```bash
pip install torch torch-geometric pymatgen matbench numpy tqdm
```

## 資料集說明

### MatBench 資料集

MatBench 是一個標準化的材料機器學習基準資料集，`matbench_mp_e_form` 任務包含：

- **訓練集**：約 106,000 個晶體結構
- **測試集**：約 26,000 個晶體結構
- **資料格式**：CIF 檔案（Crystallographic Information File）
- **目標值**：形成能（eV/atom）

### 資料集下載

資料集會在使用時自動下載（首次運行時）：

```python
from matbench import MatbenchBenchmark
mb = MatbenchBenchmark(autoload=False)
task = mb.load_task('matbench_mp_e_form')
```

### 使用子集進行快速測試

如果完整資料集太大，可以使用子集進行測試：

```bash
python experiment.py --use_subset --subset_size 1000
```

## Baseline 模型：CGCNN

### 模型架構

**CGCNN (Crystal Graph Convolutional Neural Network)** 是專門為晶體結構設計的圖神經網絡：

1. **晶體圖構建**：
   - 將晶體結構轉換為圖
   - 考慮週期性邊界條件（Periodic Boundary Conditions）
   - 使用距離截斷（cutoff）尋找鄰居原子

2. **圖卷積層**：
   - 週期性圖卷積（Periodic Graph Convolution）
   - 邊緣特徵編碼（距離、週期性影像偏移）
   - 節點特徵更新

3. **回歸頭**：
   - 全局池化（Global Mean Pooling）
   - 全連接層預測形成能

### 關鍵特性

- **週期性處理**：明確處理週期性邊界條件
- **局部特徵**：捕捉原子局部環境
- **計算效率**：線性複雜度 O(N)

### Baseline 限制

- **感受野有限**：僅考慮局部鄰居（cutoff 範圍內）
- **長程相互作用**：難以捕捉靜電、范德華力等長程效應
- **多尺度特徵**：缺乏晶體級別的特徵融合

## 執行方式

### 基本執行

```bash
python experiment.py --out_dir ./output
```

### 使用子集（快速測試）

```bash
python experiment.py --use_subset --subset_size 1000 --epochs 10
```

### 完整參數選項

```bash
python experiment.py \
    --out_dir ./output \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --hidden_dim 128 \
    --num_layers 3 \
    --cutoff 8.0 \
    --max_neighbors 12 \
    --save_model
```

### 參數說明

- `--out_dir`: 輸出目錄（預設：`./output`）
- `--batch_size`: 批次大小（預設：32）
- `--epochs`: 訓練輪數（預設：100）
- `--lr`: 學習率（預設：0.001）
- `--hidden_dim`: 隱藏層維度（預設：128）
- `--num_layers`: 圖卷積層數（預設：3）
- `--cutoff`: 鄰居距離截斷（Angstrom，預設：8.0）
- `--max_neighbors`: 每個原子的最大鄰居數（預設：12）
- `--use_subset`: 使用資料子集（用於快速測試）
- `--subset_size`: 子集大小（預設：1000）
- `--save_model`: 保存最佳模型

## 輸出結果

執行完成後，會在 `out_dir` 目錄下生成：

- `final_info.json`: 最終評估結果
  ```json
  {
      "AutoMaterial": {
          "means": {
              "MAE": "0.123456"
          }
      }
  }
  ```

- `best_model.pt`: 最佳模型權重（如果使用 `--save_model`）

## 改進方向建議

基於 baseline 的限制，可以考慮以下改進方向：

1. **長程相互作用**：
   - 注意力機制捕捉遠距離原子對
   - Ewald 求和近似靜電相互作用

2. **多尺度特徵融合**：
   - 層次化圖表示（原子 → 單元胞 → 超胞）
   - 結合晶體對稱性資訊

3. **架構改進**：
   - 3-body 相互作用（如 M3GNet）
   - 更深的網絡或殘差連接
   - 圖注意力機制

4. **特徵工程**：
   - 更豐富的原子特徵（電負性、離子半徑等）
   - 晶體描述符（空間群、密度等）

## 參考文獻

- **CGCNN**: Xie, T., & Grossman, J. C. (2018). Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. *Physical review letters*, 120(14), 145301.

- **MatBench**: Dunn, A., et al. (2020). Benchmarking materials property prediction methods: the Matbench test set for automated machine learning. *npj Computational Materials*, 6(1), 138.

- **Materials Project**: Jain, A., et al. (2013). Commentary: The Materials Project: A materials genome approach to accelerating materials innovation. *APL materials*, 1(1).

## 常見問題

### Q: 資料集下載失敗怎麼辦？

A: 確保網路連線正常，或手動下載 MatBench 資料集。也可以使用 `--use_subset` 選項進行測試。

### Q: 記憶體不足？

A: 減少 `--batch_size` 或使用 `--use_subset` 選項。

### Q: 訓練速度太慢？

A: 使用 GPU（如果可用），或減少 `--subset_size` 和 `--epochs`。

### Q: 如何載入已保存的模型？

A: 在 `experiment.py` 中添加模型載入邏輯：
```python
model.load_state_dict(torch.load('best_model.pt'))
```

## 聯絡與支援

如有問題或建議，請參考專案主 README 或提交 issue。

