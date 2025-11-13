# AutoMaterial 案例實作計劃

## 📋 Todo List 詳細說明

### ✅ Todo 1: 深入分析 MatBench 資料集

**目標**：全面了解 `matbench_mp_e_form` 資料集的特性

**具體工作**：
1. **資料格式分析**
   - CIF 檔案結構（Crystallographic Information File）
   - 形成能（Formation Energy）的數值範圍和分佈
   - 訓練/測試分割方式（MatBench 標準協議）

2. **統計資訊收集**
   - 樣本數量：132,752 個
   - 元素種類和分佈
   - 晶體結構類型（空間群、對稱性）
   - 形成能的分佈統計（均值、標準差、分位數）

3. **評估協議確認**
   - 評估指標：MAE（Mean Absolute Error）
   - 交叉驗證方式（如有）
   - 基準模型性能（CGCNN、M3GNet 等）

4. **技術挑戰識別**
   - 週期性結構表示需求
   - 長程相互作用的重要性
   - 資料不平衡問題（如有）

**輸出**：
- 資料集分析報告（整合到 prompt.json 的 background 中）
- 資料集下載和預處理指南

---

### ✅ Todo 2: 設計 prompt.json

**目標**：創建符合 InternAgent 規範的任務描述檔案

**檔案結構**：
```json
{
    "system": "...",
    "task_description": "...",
    "domain": "materials property prediction",
    "background": "...",
    "constraints": [...]
}
```

**具體內容設計**：

1. **system**：
   - 定位為材料科學 AI 研究員
   - 專注於晶體結構性質預測

2. **task_description**：
   - 明確任務：根據晶體結構預測形成能
   - 輸入格式：CIF 檔案（原子類型、位置、晶格參數）
   - 輸出：形成能（eV/atom，回歸任務）
   - 評估資料集：MatBench matbench_mp_e_form
   - 評估指標：MAE

3. **domain**：
   - "materials property prediction" 或 "crystal structure property prediction"

4. **background**（詳細技術背景）：
   - **技術挑戰**：
     - 週期性結構表示（Periodic Boundary Conditions）
     - 長程相互作用（靜電、范德華力）
     - 多尺度特徵融合（原子級到晶體級）
   - **Baseline 方法**：
     - CGCNN（Crystal Graph Convolutional Neural Network）
     - M3GNet（Materials 3-body Graph Network）
     - SchNet（Schütt et al.）
   - **Baseline 限制**：
     - 感受野有限（難以捕捉長程相互作用）
     - 特徵表示可能不夠豐富
     - 多尺度特徵融合不足

5. **constraints**：
   - 必須使用晶體結構作為輸入（CIF 格式）
   - 評估指標固定為 MAE
   - 必須處理週期性邊界條件
   - 模型改進需專注於架構設計

**參考格式**：
- 參考 `AutoChem/prompt.json` 的詳細程度
- 參考 `AutoMolecule3D/prompt.json` 的技術深度

---

### ✅ Todo 3: 設計 baseline experiment.py

**目標**：創建可執行的 baseline 實作

**檔案結構**：

1. **導入與依賴**：
   ```python
   import torch
   import torch.nn as nn
   from torch_geometric.data import Data, Batch
   from pymatgen.core import Structure
   from matbench import MatbenchBenchmark
   # ... 其他依賴
   ```

2. **資料載入模組**：
   - MatBench 資料集載入
   - CIF 檔案解析（使用 pymatgen）
   - 晶體圖構建（考慮週期性鄰居）
   - 資料預處理和標準化

3. **模型架構**：
   - **Baseline 選擇**：CGCNN 或 M3GNet
   - 週期性圖卷積層
   - 特徵聚合層
   - 回歸頭（預測形成能）

4. **訓練循環**：
   - 損失函數：MSE Loss
   - 優化器：Adam
   - 學習率調度
   - 訓練/驗證循環
   - 模型保存

5. **評估模組**：
   - MAE 計算
   - 結果記錄（JSON 格式）
   - 可選：視覺化（預測 vs 真實值）

6. **主函數**：
   - 參數解析（argparse）
   - 資料載入
   - 模型初始化
   - 訓練和評估
   - 結果輸出

**Baseline 模型選擇**：
- **CGCNN**：經典的晶體圖神經網絡，實作相對簡單
- **M3GNet**：更先進，包含 3-body 相互作用
- **建議**：先實作 CGCNN，作為 baseline

**輸出格式**：
- 符合 InternAgent 的結果格式要求
- 生成 `final_info.json` 包含 MAE 結果

---

### ✅ Todo 4: 創建任務目錄結構

**目標**：建立完整的任務目錄和檔案

**目錄結構**：
```
tasks/AutoMaterial/
├── prompt.json          # 任務描述（Todo 2）
├── experiment.py       # Baseline 實作（Todo 3）
├── launcher.sh         # 可選：執行腳本
├── README.md           # 可選：任務說明
└── run_0/              # Baseline 運行結果（可選）
    └── experiment.py   # Baseline 版本（與根目錄相同或簡化版）
```

**檔案說明**：
1. **prompt.json**：主要任務描述檔案（必須）
2. **experiment.py**：可執行的 baseline 實作（必須）
3. **launcher.sh**：方便執行的腳本（可選，參考其他任務）
4. **README.md**：資料集下載、環境設置說明（可選但建議）

---

### ✅ Todo 5: 撰寫資料集下載與預處理說明

**目標**：提供完整的資料集使用指南

**內容包括**：

1. **MatBench 安裝**：
   ```bash
   pip install matbench
   ```

2. **資料集下載**：
   ```python
   from matbench import MatbenchBenchmark
   mb = MatbenchBenchmark(autoload=False)
   mb.load_task("matbench_mp_e_form")
   ```

3. **資料格式說明**：
   - CIF 檔案格式
   - 形成能單位（eV/atom）
   - 訓練/測試分割方式

4. **環境依賴**：
   - PyTorch
   - PyTorch Geometric
   - pymatgen
   - matbench
   - 其他必要套件

5. **預處理步驟**：
   - CIF 檔案解析
   - 晶體圖構建
   - 特徵標準化

**輸出位置**：
- 可整合到 `tasks/AutoMaterial/README.md`
- 或在 `prompt.json` 的 background 中簡要說明

---

### ✅ Todo 6: 驗證案例完整性

**目標**：確保新案例符合 InternAgent 規範

**驗證項目**：

1. **檔案格式驗證**：
   - `prompt.json` 格式正確（JSON 驗證）
   - 包含所有必要欄位（system, task_description, domain, background, constraints）

2. **程式碼驗證**：
   - `experiment.py` 語法正確
   - 可以成功導入（無 import 錯誤）
   - 基本執行流程完整（即使資料未下載也能通過語法檢查）

3. **符合專案規範**：
   - 目錄結構符合其他任務的格式
   - 命名規範一致（AutoMaterial）
   - 輸出格式符合要求（final_info.json）

4. **文檔完整性**：
   - prompt.json 描述清晰
   - 技術背景充分
   - 約束條件明確

**驗證方法**：
- 使用 `launch_discovery.py` 測試任務載入
- 檢查 prompt.json 是否能被正確解析
- 驗證 experiment.py 的基本結構

---

## 🎯 實作順序建議

1. **Todo 1** → 深入分析資料集（為後續設計提供基礎）
2. **Todo 2** → 設計 prompt.json（核心任務描述）
3. **Todo 3** → 設計 experiment.py（baseline 實作）
4. **Todo 4** → 創建目錄結構（組織檔案）
5. **Todo 5** → 撰寫說明文件（完善文檔）
6. **Todo 6** → 驗證完整性（品質保證）

---

## 📝 技術決策點（需討論確認）

### 1. Baseline 模型選擇
- **選項 A**：CGCNN（經典、簡單、易實作）
- **選項 B**：M3GNet（先進、包含 3-body 相互作用）
- **建議**：先實作 CGCNN，後續可擴展到 M3GNet

### 2. 資料集規模
- **完整資料集**：132,752 樣本（可能計算量大）
- **子集**：可先使用較小規模進行測試
- **建議**：實作時支援完整資料集，但提供選項使用子集

### 3. 評估指標
- **主要指標**：MAE（MatBench 標準）
- **可選指標**：RMSE、R²（用於分析，不作為主要評估）

### 4. 任務命名
- **選項 A**：`AutoMaterial`（簡潔）
- **選項 B**：`AutoMP`（Materials Property 縮寫）
- **選項 C**：`AutoFormE`（Formation Energy 縮寫）
- **建議**：`AutoMaterial`（最直觀）

---

## ❓ 需要確認的問題

1. **Baseline 模型**：選擇 CGCNN 還是 M3GNet？
2. **資料集規模**：是否先使用完整資料集，還是提供子集選項？
3. **任務命名**：確認使用 `AutoMaterial` 還是其他名稱？
4. **技術深度**：prompt.json 的 background 需要多詳細？（參考 AutoMolecule3D 的深度？）
5. **依賴管理**：是否需要 requirements.txt 或環境說明？

---

## 📦 預期輸出

完成後將得到：
- ✅ `tasks/AutoMaterial/prompt.json` - 完整的任務描述
- ✅ `tasks/AutoMaterial/experiment.py` - 可執行的 baseline 實作
- ✅ `tasks/AutoMaterial/README.md` - 資料集和環境說明（可選）
- ✅ 資料集分析報告（整合在 prompt.json 中）

---

## 🚀 下一步

請確認：
1. Todo list 是否完整？
2. 技術決策點是否有異議？
3. 是否需要調整實作順序？
4. 是否有其他特殊要求？

確認後即可開始實作！

