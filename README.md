# 混米辨別與混米比例估計（MRPM）
# Mixed Rice Detection & Mixing-Ratio Prediction (MRPM)

## 專案簡介 / Overview
**中文：**  
本專案旨在辨別樣本是否為「混米」（摻入不同來源的米），並在判定為混米時估計混入比例（%）。流程包含：以純米資料生成混米樣本、訓練與比較多種分類模型（Taiwan / Vietnam / Mixed 三分類）、以及建立混米比例的回歸模型，最後整合為可使用的 **MRPM** 模型。

**English:**  
This project detects whether a rice sample is *mixed/adulterated* and, if mixed, estimates the mixing ratio (%). The pipeline includes: generating synthetic mixed samples from pure rice data, training & comparing multiple classifiers for a 3-class task (Taiwan / Vietnam / Mixed), fitting regression models for mixing-ratio estimation, and integrating them into a usable **MRPM** model.

---

## 功能特色 / Key Features
**中文：**
- 由純米資料合成混米資料（可控比例）
- 比較多種分類器的準確率（Accuracy）
- 混米比例（ratio_t / ratio_v）估計與 95% 信賴區間
- MRPM 整合輸出：回傳表格，方便匯出 CSV

**English:**
- Synthetic mixed-sample generation from pure rice samples (controllable ratios)
- Accuracy comparison across multiple classifiers
- Mixing-ratio prediction (ratio_t / ratio_v) with 95% confidence intervals
- MRPM integrated output as a table for easy CSV export

---

## 資料說明 / Data Description
**中文：**
- 原始資料需包含：
  - `Country`：至少包含 `Taiwan`、`Vietnam`（混米資料會在程式中標記為 `Mixed`）
  - 特徵欄位：本專案使用 `X1` ~ `X5`（你的程式以欄位位置 `2:6` 讀取特徵）
- 混米比例欄位：
  - `ratio_t`：台灣米比例
  - `ratio_v`：越南米比例（= 1 - ratio_t）

**English:**
- The raw dataset should contain:
  - `Country`: at least `Taiwan` and `Vietnam` (synthetic mixed samples will be labeled as `Mixed`)
  - Feature columns: `X1` ~ `X5` (the script uses columns `2:6` as features)
- Ratio columns:
  - `ratio_t`: Taiwan proportion
  - `ratio_v`: Vietnam proportion (= 1 - ratio_t)

---

## 方法概述 / Method Summary

### 1) 混米資料生成 / Mixed-sample Generation
**中文：**  
使用純台灣米與純越南米樣本，隨機抽樣並以線性混合方式生成混米特徵：  
`mixed = vietnam * ratio_v + taiwan * ratio_t`  
其中 `ratio_t` 隨機從 0.1～0.9（步長 0.1）抽取，`ratio_v = 1 - ratio_t`。

**English:**  
Synthetic mixed samples are created by randomly selecting one Taiwan sample and one Vietnam sample, then linearly mixing features:  
`mixed = vietnam * ratio_v + taiwan * ratio_t`  
where `ratio_t` is randomly sampled from 0.1 to 0.9 (step 0.1), and `ratio_v = 1 - ratio_t`.

> 注意 / Note: 這是「線性混合」的合成資料假設，適用於特徵可近似線性疊加的情境。

---

### 2) 分類任務 / Classification Task
**中文：**  
建立三分類資料集：`Taiwan / Vietnam / Mixed`，並訓練比較多種分類模型（例如 LDA、PLS、Ridge、LASSO、Random Forest、SVM），以 **Accuracy** 為主要指標評估訓練集與測試集表現。

**English:**  
A 3-class classification dataset is built: `Taiwan / Vietnam / Mixed`. Multiple classifiers (e.g., LDA, PLS, Ridge, LASSO, Random Forest, SVM) are trained and evaluated using **Accuracy** on both train and test sets.

---

### 3) 混米比例估計 / Mixing-ratio Estimation
**中文：**  
另外生成較大量混米資料（如 1000 筆）用於回歸模型訓練，以特徵 `X1~X5` 預測：
- `ratio_t`（台灣比例）
- `ratio_v`（越南比例）  
並以 MSE 評估誤差，並可輸出 95% 信賴區間。

**English:**  
A larger mixed dataset (e.g., 1000 samples) is generated for regression. Using `X1~X5`, the model predicts:
- `ratio_t` (Taiwan ratio)
- `ratio_v` (Vietnam ratio)  
MSE is used for evaluation, and 95% confidence intervals can be produced.

---

## MRPM：整合模型 / Integrated MRPM
**中文：**  
MRPM 將「分類」與「比例估計」串接：
1. 先用分類器預測 `Taiwan / Vietnam / Mixed`
2. 若預測為混米，則使用回歸模型估計混入比例與 95% CI  
在你的現行程式中，MRPM 使用 **polynomial kernel 的 SVM** 作分類，並使用 **線性回歸（lm）** 做比例估計；最終回傳表格，可直接匯出 CSV。

**English:**  
MRPM chains classification and ratio prediction:
1. Predict `Taiwan / Vietnam / Mixed` using a classifier
2. If predicted as `Mixed`, estimate mixing ratios with confidence intervals  
In the current script, MRPM uses **polynomial-kernel SVM** for classification and **linear regression (lm)** for ratio estimation. It returns a table suitable for CSV export.

---

## 使用方式 / Usage

### 需求套件 / Dependencies
**中文：** R 套件：`caret`, `MASS`, `pls`, `glmnet`, `randomForest`, `e1071`  
**English:** R packages: `caret`, `MASS`, `pls`, `glmnet`, `randomForest`, `e1071`

### 執行流程 / Typical Workflow
**中文：**
1. 準備 `raw.data`（包含 `Country` 與 `X1~X5`）
2. 執行腳本：生成混米、切分資料、訓練並比較分類器
3. 生成大量混米樣本訓練比例回歸模型
4. 呼叫 `MRPM(newdata)` 輸出預測表格並匯出 CSV

**English:**
1. Prepare `raw.data` (with `Country` and `X1~X5`)
2. Run the script to generate mixed samples, split data, train & compare classifiers
3. Generate a larger mixed set to train regression models for ratios
4. Call `MRPM(newdata)` to obtain a prediction table and export it as CSV

---

## 輸出格式 / Output Format
MRPM（表格版）輸出欄位 / MRPM (table version) columns:
- `Actual`: 真實類別（若 newdata 有提供）/ Ground truth label (if available in input)
- `Predict`: 預測類別（Taiwan / Vietnam / Mix）/ Predicted label (Taiwan / Vietnam / Mix)
- `Ratio`:（目前輸出）越南比例點估計 / (currently) point estimate of Vietnam ratio
- `CI`: 95% 信賴區間字串 / 95% CI as a string

> 備註 / Note: 你目前表格版 MRPM 只輸出 `ratio_v`（越南比例）。若需要同時輸出 `ratio_t`，可在未來擴充。

---

## 重要假設與限制 / Assumptions & Limitations
**中文：**
- 混米資料為合成資料：特徵以線性混合生成，未必完全反映真實混米的物理/化學過程
- 分類器比較使用 Accuracy；若資料不平衡，建議額外報告 F1、balanced accuracy 等指標
- MRPM 的「最佳模型」目前為人工指定（程式中用 polynomial SVM + lm），未自動依比較結果選出

**English:**
- Mixed samples are synthetic (linear mixing), which may not fully reflect real-world mixing processes
- Accuracy is the primary metric; for imbalanced data, consider reporting F1 / balanced accuracy
- The “best” MRPM is currently chosen manually in the script (polynomial SVM + lm), not automatically selected based on comparison results

---

## 專案結構（建議）/ Suggested Repo Structure
```text
.
├─ data/            # raw data (optional; avoid committing large files)
├─ scripts/         # R scripts
├─ results/         # exported predictions / figures
├─ README.md
└─ LICENSE
