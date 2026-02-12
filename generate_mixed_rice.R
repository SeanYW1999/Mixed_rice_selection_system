## =============================================================================
## 混米辨識與混米比例估計（Taiwan / Vietnam / Mixed）
## Rice Adulteration Detection & Mixing-Ratio Estimation (Taiwan / Vietnam / Mixed)
##
## 流程概覽 / Pipeline Overview:
##  1) 讀入原始資料並檢視欄位 / Load data & inspect structure
##  2) 取出純米並加上比例標記 / Extract pure samples & add ratio labels
##  3) 生成混米樣本 / Generate synthetic mixed samples
##  4) 建立分類資料集並切分 / Build classification dataset & split train/test
##  5) 比較多種分類模型 / Compare multiple classifiers
##  6) 建立比例回歸模型 / Train regression models for mixing ratio
##  7) 整合模型 MRPM：分類 + 比例估計 / Integrated MRPM: classification + ratio estimation
## =============================================================================

raw.data <- read.csv()
str(raw.data[, 2:8])
summary(raw.data[, 2:8])

## ---------------------------
## 評估指標：Accuracy（準確率）
## Metric: Accuracy
## - 輸入：混淆矩陣 / Input: confusion matrix
## - 輸出：印出準確率 / Output: print accuracy
## ---------------------------
cfmaccuracy <- function(matrix){
  
  accuracy <- sum(diag(matrix)) / sum(matrix)
  return(cat("準確率：", accuracy, "\n"))
  
}

## ---------------------------
## 資料清理：抽取純米（越南/台灣）並建立比例欄位
## Data prep: extract pure samples (Vietnam/Taiwan) & create ratio columns
## - 純越南米：ratio_v=1, ratio_t=0 / Pure Vietnam: ratio_v=1, ratio_t=0
## - 純台灣米：ratio_t=1, ratio_v=0 / Pure Taiwan: ratio_t=1, ratio_v=0
## ---------------------------
vietnam <- raw.data[which(raw.data$Country == "Vietnam"), ] # 抽取 rawdata 中的越南樣本 / Filter Vietnam samples
v <- cbind(vietnam[,2:7], ratio_t = 0, ratio_v = 1)         # 純越南米 / Pure Vietnam

taiwan <- raw.data[which(raw.data$Country == "Taiwan"), ]   # 抽取 rawdata 中的臺灣樣本 / Filter Taiwan samples
t <- cbind(taiwan[,2:7], ratio_t = 1, ratio_v = 0)          # 純台灣米 / Pure Taiwan

### ---------------------------
### 創建混米生成函式
### Create a synthetic mixed-sample generator
### - 每次隨機抽一筆台灣與越南樣本 / Randomly pick one Taiwan & one Vietnam sample each time
### - 隨機設定台灣比例 0.1~0.9 / Randomly set Taiwan ratio from 0.1 to 0.9
### - 特徵以線性加權混合 / Linearly mix feature values by the chosen ratios
### - 標記 Country="Mixed" 並附上 ratio_t/ratio_v / Label Country="Mixed" and attach ratios
### ---------------------------
mixedsample <- function(data, times){
  mixed_data <- data.frame() # 創建空的 dataframe 儲存mixeddata / Initialize empty df to store mixed samples
  for (i in 1:times) {
    vietnam_sample <- vietnam[sample(1:70, 1), ] #隨機抽取 rawdata 中的越南米 / Randomly sample a Vietnam instance
    taiwan_sample <- taiwan[sample(1:80, 1), ]   #隨機抽取 rawdata 中的臺灣米 / Randomly sample a Taiwan instance
    
    ratio_t <- sample(1:9, 1)/10 #隨機產生混米中臺灣米的比例 / Random Taiwan proportion in mixture
    ratio_v <- 1-ratio_t         #越南米的比例 / Vietnam proportion
    
    vietnam_values <- as.numeric(vietnam_sample[, 2:6]) 
    taiwan_values <- as.numeric(taiwan_sample[, 2:6])
    
    #計算混米比例（特徵線性混合）/ Compute mixed features (linear combination)
    mixed_values <- round(vietnam_values * ratio_v + taiwan_values * ratio_t, 3) 
    
    mixed_sample <- data.frame(t(mixed_values))
    
    #將混米的國家標記為mixed並且將比例的cbind上去 / Set Country="Mixed" and bind ratios
    mixed_sample <- cbind(mixed_sample, Country = "Mixed", ratio_t, ratio_v)
    #把純米跟混米資料結合成同一個資料框 / Append to mixed_data
    mixed_data <- rbind(mixed_data, mixed_sample)
  }
  return(mixed_data)
}

## ---------------------------
## 生成混米資料並建立分類資料集
## Generate mixed samples and build classification dataset
## - 混米 150 筆 / 150 mixed samples
## - 合併純米與混米 / Combine pure + mixed
## - 80% 訓練 / 20% 測試 / 80% train / 20% test
## ---------------------------
m <- mixedsample(raw.data, 150) #生成150筆混米資料 / Generate 150 mixed samples
data.clean <- rbind(t, v, m) 
rownames(data.clean) <- NULL

train_index <- sample(1:nrow(data.clean), 0.8 * nrow(data.clean)) # 隨機抽建立訓練集編號 / Random train indices (80%)

train_data <- data.clean[train_index, ]
test_data <- data.clean[-train_index, ]
train_data$Country <- as.factor(train_data$Country)
test_data$Country <- as.factor(test_data$Country)

#############
#### LDA ####
#############
## 線性判別分析分類 / Classification via Linear Discriminant Analysis

library(caret)
library(MASS)

#設定cv的控制參數 / CV control settings
cv_control <- trainControl(method = "cv", number = 10) # 10-fold

lda_model <- train(Country ~ X1 + X2 + X3 + X4 + X5, 
                      data = train_data, 
                      method = "lda", 
                      trControl = cv_control)

##訓練集 / Train set
LDA_fit <- predict(lda_model, newdata = train_data)
lda_cm <- table(predict = LDA_fit, actual = train_data$Country)
cfmaccuracy(lda_cm)

##測試集 / Test set
LDA_fit <- predict(lda_model, newdata = test_data)
lda_cm <- table(predict = LDA_fit, actual = test_data$Country)
cfmaccuracy(lda_cm)

#############
#### PLS ####
#############
## 偏最小平方（PLS）多分類（one-hot + argmax）/ PLS for multiclass (one-hot + argmax)

library(pls)
train_response <- model.matrix(~ Country - 1, data = train_data)
test_response <- model.matrix(~ Country - 1, data = test_data)

# 執行 PLS 分析 / Fit PLS model
pls_model <- plsr(train_response ~ X1 + X2 + X3 + X4 + X5, 
                  data = train_data, scale = TRUE, validation = "CV", 
                  segments = 10)

summary(pls_model)

validationplot(pls_model, val.type = "MSEP")

## 計算不同 ncomp 下的分類準確率並回報最佳值
## Compute accuracy across ncomp and report the best setting
pls_accuracy <- function(model,data){
  result.a <- c()
  for (ncomp in 1:5) {
  pls_fit <- predict(model, newdata = data, ncomp = ncomp)
  # 轉換為分類結果 / Convert predicted scores to class labels
  # 找到每行的最大值索引 / Argmax for each row
  pls_class <- apply(pls_fit, 1, which.max)
  
  # 將索引轉換為原始類別標籤 / Map indices back to class names
  levels <- colnames(train_response)  # 原始標籤名稱 / class names
  pls_class <- factor(levels[pls_class], levels = levels)

  # 建立混淆矩陣並計算準確率 / Confusion matrix and accuracy
  pls_cm <- table(Predicted = pls_class, Actual = data$Country)
  accuracy <- round(sum(diag(pls_cm)) / sum(pls_cm),3)
  result.a[ncomp] <- accuracy
  }
  best.ncomp <- which(result.a == max(result.a))
  best.acc <- max(result.a)
  return(cat("最佳變數數量：", best.ncomp,"\n","準確率：", best.acc))
} 

pls_accuracy(pls_model,train_data)
pls_accuracy(pls_model,test_data)

##########################
#### Ridge Regression ####
##########################
## Ridge (multinomial) 分類 / Multinomial Ridge classification

library(glmnet)

train_X <- as.matrix(train_data[, c("X1", "X2", "X3", "X4", "X5")])
test_X <- as.matrix(test_data[, c("X1", "X2", "X3", "X4", "X5")])
train_Y <- as.factor(train_data$Country)
test_Y <- as.factor(test_data$Country)

# 使用 glmnet 進行 Ridge Regression / Ridge via glmnet
ridge_model <- cv.glmnet(train_X, train_Y, 
                         alpha = 0,               # alpha = 0 表示 Ridge Regression / Ridge
                         family = "multinomial",  # 多分類 / multiclass
                         nfolds = 10)             # 10-fold CV

# 顯示 交叉驗證誤差最小 的 lambda 值 / Best lambda (min CV error)
optimal_lambda <- ridge_model$lambda.min
plot(ridge_model)
cat("最佳的 Lambda 值：", optimal_lambda, "\n")

### 訓練集 / Train set ###
ridge_fit <- predict(ridge_model, newx = train_X, s = optimal_lambda, 
                     type = "class") 
ridge_cm <- table(Predicted = ridge_fit, Actual = train_data$Country)
cfmaccuracy(ridge_cm)

### 測試集 / Test set ###
ridge_fit <- predict(ridge_model, newx = test_X, s = optimal_lambda, 
                        type = "class")
ridge_cm <- table(Predicted = ridge_fit, Actual = test_data$Country)
cfmaccuracy(ridge_cm)

###############
#### LASSO ####
###############
## LASSO (multinomial) 分類 / Multinomial LASSO classification

# 使用 glmnet 進行 LASSO Regression / LASSO via glmnet
lasso_model <- cv.glmnet(train_X, train_Y, 
                         alpha = 1,               # alpha = 1 表示 LASSO / LASSO
                         family = "multinomial",  # 多分類 / multiclass
                         nfolds = 10)             # 10-fold CV

# optimal lambda / Best lambda
optimal_lambda <- lasso_model$lambda.min
plot(lasso_model)
cat("最佳的 Lambda 值：", optimal_lambda, "\n")

### 訓練集 / Train set ###
lasso_fit <- predict(lasso_model, newx = train_X, s = optimal_lambda, 
                     type = "class")
lasso_cm <- table(Predicted = lasso_fit, Actual = train_data$Country)
cfmaccuracy(lasso_cm)

### 測試集 / Test set ###
lasso_fit <- predict(lasso_model, newx = test_X, s = optimal_lambda, 
                        type = "class")
lasso_cm <- table(Predicted = lasso_fit, Actual = test_data$Country)
cfmaccuracy(lasso_cm)

#######################
#### Random Forest ####
#######################
## 隨機森林分類 / Random Forest classifier

library(randomForest)

rf_model <- randomForest(as.factor(Country) ~ X1 + X2 + X3 + X4 + X5, 
                         data = train_data,
                         ntree = 500,        # 樹的數量 / number of trees
                         mtry = sqrt(5),     # 每次分割時的特徵數 / features per split
                         importance = TRUE)  # 特徵重要性 / feature importance

plot(rf_model)

## 訓練集 / Train set
rf_fit <- predict(rf_model, newdata = train_data)
RF_cm <- table(Predicted = rf_fit, Predicted = train_data$Country) # 註：欄位名重複，程式不改 / Note: duplicated name, code unchanged
print(RF_cm)
cfmaccuracy(RF_cm)

## 測試集 / Test set
rf_fit <- predict(rf_model, newdata = test_data)
RF_cm <- table(Predicted = rf_fit, Predicted = test_data$Country)  # 註：欄位名重複，程式不改 / Note: duplicated name, code unchanged
print(RF_cm)
cfmaccuracy(RF_cm)


#############
#### SVM ####
#############
## 支援向量機分類 / Support Vector Machine classifiers

library(e1071)

# 查看最佳參數 cost / Tune best cost for a given kernel
best.cost <- function(kernel){
  tune_result <- tune(svm, Country ~  X1 + X2 + X3 + X4 + X5, 
  data = train_data, 
  ranges = list(cost = c(0.001 , 0.01, 0.1, 1,5,10,100)),
  kernel = kernel
)
  return(tune_result$best.parameters)
}

# 訓練 SVM 模型（不同 kernel）/ Train SVM models with different kernels
### linear ###
svm_model_l <- svm(Country ~  X1 + X2 + X3 + X4 + X5, 
                 data = train_data, 
                 kernel = "linear",
                 cost = best.cost("linear"),
                 scale = TRUE)

### radial ###
svm_model_r <- svm(Country ~  X1 + X2 + X3 + X4 + X5, 
                   data = train_data, 
                   kernel = "radial",  
                   cost = best.cost("radial"),           
                   scale = TRUE)
### polynomial ###
svm_model_p <- svm(Country ~  X1 + X2 + X3 + X4 + X5, 
                   data = train_data, 
                   kernel = "polynomial",  
                   cost = best.cost("polynomial"),           
                   scale = TRUE)

### sigmoid ###
svm_model_s <- svm(Country ~  X1 + X2 + X3 + X4 + X5, 
                   data = train_data, 
                   kernel = "sigmoid",  
                   cost = best.cost("sigmoid"),           
                   scale = TRUE)

# 在測試集上進行預測 / Predict on test set
svm_fit_l <- predict(svm_model_l, newdata = test_data)
svm_fit_r <- predict(svm_model_r, newdata = test_data)
svm_fit_p <- predict(svm_model_p, newdata = test_data)
svm_fit_s <- predict(svm_model_s, newdata = test_data)

svm_cm_l <- table(Predicted = svm_fit_l, Actual = test_data$Country)
svm_cm_r <- table(Predicted = svm_fit_r, Actual = test_data$Country)
svm_cm_p <- table(Predicted = svm_fit_p, Actual = test_data$Country)
svm_cm_s <- table(Predicted = svm_fit_s, Actual = test_data$Country)

cfmaccuracy(svm_cm_l)
cfmaccuracy(svm_cm_r)
cfmaccuracy(svm_cm_p)
cfmaccuracy(svm_cm_s)

#########################################################################################
#########################################################################################
#########################################################################################

#######################
####混米比例估計模型####
#######################
## Mixing-ratio estimation model (regression on mixed samples)

m2 <- mixedsample(raw.data, 1000) # 生成 1000 筆混米 / Generate 1000 mixed samples
train_index <- sample(1:nrow(m2), 0.8 * nrow(m2)) # 隨機抽建立訓練集編號 / Train indices (80%)
train_data2 <- m2[train_index, ]
test_data2 <- m2[-train_index, ]

#### Linear Models / 線性回歸模型

lm_model_t <- lm(ratio_t ~ X1 + X2 + X3 + X4 + X5, data = train_data2) # 預測台灣比例 / Predict Taiwan ratio
lm_model_v <- lm(ratio_v ~ X1 + X2 + X3 + X4 + X5, data = train_data2) # 預測越南比例 / Predict Vietnam ratio

## 訓練集 MSE / Train MSE
ratio_t_pred1 <- predict(lm_model_t, newdata = train_data2)
ratio_v_pred1 <- predict(lm_model_v, newdata = train_data2)
mean((ratio_t_pred1 - train_data2$ratio_t)^2)
mean((ratio_v_pred1 - train_data2$ratio_v)^2)

## 測試集 MSE / Test MSE
ratio_t_pred2 <- predict(lm_model_t, newdata = test_data2)
ratio_v_pred2 <- predict(lm_model_v, newdata = test_data2)
mean((ratio_t_pred2 - test_data2$ratio_t)^2)
mean((ratio_v_pred2 - test_data2$ratio_v)^2)

## 95% 信賴區間（confidence interval）/ 95% confidence intervals
ratio_t_pred_with_ci <- predict(lm_model_t, newdata = test_data2, 
                                interval = "confidence", level = 0.95)
ratio_v_pred_with_ci <- predict(lm_model_v, newdata = test_data2, 
                                interval = "confidence", level = 0.95)

## ---------------------------
## 結合鑑別與比例估計（基礎版：文字輸出）
## Combine classification & ratio estimation (basic: console text output)
## - 先用 polynomial SVM 判斷 Mixed / First classify with polynomial SVM
## - 只對判定為 Mixed 的樣本估比例 / Estimate ratios only for predicted Mixed samples
## - 輸出：逐筆列印 + 總混米筆數 / Output: per-sample print + total count
## ---------------------------
MRPM <- function(newdata){
  svm_p <- predict(svm_model_p, newdata = newdata)
  input <- newdata[which(svm_p == "Mixed"),]

  t <- predict(lm_model_t, newdata = input, 
             interval = "confidence", level = 0.95)
  t <- round(t,3)

  v <- predict(lm_model_v, newdata = input, 
             interval = "confidence", level = 0.95)
  v <- round(v,3)
  row.name <- as.numeric(row.names(input))
  
  for (n in 1:nrow(input)) {
    cat("第",row.name[n],"筆資料是混米","\n",
        "台灣米比例為:",t[n,1]," 95% 信賴區間為:",t[n,2:3],"\n",
        "越南米比例為:",v[n,1]," 95% 信賴區間為:",v[n,2:3],"\n","\n")
  }
  return(cat("有",nrow(input),"筆資料是混米"))
}

## ---------------------------
## 結合鑑別與比例估計（進階版：輸出表格）
## Combine classification & ratio estimation (advanced: return a table)
## 注意：此函式會覆蓋上一個 MRPM / Note: this MRPM overwrites the previous MRPM
## - 非 Mixed：Ratio/CI 為 NA / Non-mixed: Ratio/CI are NA
## - Mixed：輸出越南比例 ratio_v 及 95% CI / Mixed: output Vietnam ratio_v and 95% CI
## - 回傳 data.frame 方便 View()/write.csv() / Returns a data.frame for viewing/export
## ---------------------------
MRPM <- function(newdata){
  pt <- data.frame(Actual = rep(NA,nrow(newdata)),
                   Predict = rep(NA,nrow(newdata)),
                   Ratio = rep(NA,nrow(newdata)),
                   CI = rep(NA,nrow(newdata)))
  
  svm_p <- predict(svm_model_p, newdata = newdata)
  row.name <- as.numeric(row.names(newdata))
  mix <- c()
  
  v <- predict(lm_model_v, newdata = newdata, 
               interval = "confidence", level = 0.95)
  v <- round(v,3)
  
  for (n in 1:nrow(newdata)) {
    if (svm_p[n] == "Taiwan"){
      pt[n,] <- c(newdata$Country[n],"Taiwan",NA,NA)
      
    }else if (svm_p[n] == "Vietnam"){
      pt[n,] <- c(newdata$Country[n],"Vietnam",NA,NA)
    }else {
      mix <- rbind(mix,n)
      pt[n,] <- c(Actual = newdata$Country[n], Predict = "Mix",
              Ratio = v[n,1], CI = paste("[",v[n,2],",",v[n,3],"]"))
    }
  }
  return(pt)
}

## ---------------------------
## 執行整合模型並輸出結果 / Run MRPM and export results
## ---------------------------
result <- MRPM(dat)
View(result)
write.csv(result,".csv")


 
