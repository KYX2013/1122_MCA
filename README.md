# 1122_MCA_README
## HW2 - Video Shot Change Detection
**目標：自行撰寫程式完成video shot change detection**

資料：共三部影片及其對應的video frames、shot change boundary。你的程式可選用直接輸入影片檔(mpg)或video frames。包含偵測效能（三部影片效能分別詳列）

除了課堂中講述的方法之外，盡可能找尋資料或思考實作更好的features或演算法

Datasets:
1. [news](https://youtu.be/_HrEg0BE-G0)
2. [ngc](https://youtu.be/jwkzthLvn-0)
3. [climate](https://youtu.be/u1LSGDTdIO4)

## HW4 - Music Genre Classification
**目標：自行撰寫程式完成music genre classification**

資料：來自George Tzanetakis經典論文，共10類音樂類型。為了減輕作業負擔，我將每一類減少至50個音樂片段，每個片段30秒。

實驗規定：

若你採用learning algorithms (generative model or discriminative)，請一致用5-fold cross validation。也就是說，每一類隨機選40個片段作training data，10個片段作testing data。training/testing過程進行五次後，將分類的準確率平均。

除了課堂中講述的方法之外，盡可能找尋資料或思考實作更好的features或演算法。若你使用的是deep learning方法，不見得能明確講出feature的意義，則只需詳述deep neural networks的細節。

## HW5 - CNN classifier
Problem
為了讓各位同學學習基礎 CNN 的架構，因此本次作業要求同學
1. ⾃⼰建⽴ CNN 模型
2. 訓練模型在 Dog-Cat-Pandas 分類任務上
3. 根據訓練成果改進模型

Tutorial
1. 建⽴模型
本次要求同學 使⽤ Pytorch 來建⽴深度學習模型，可以參考 AlexNet 架構，透過數
層 CNN 提取圖⽚特徵後，經過 FC 來進⾏分類。
2. 訓練模型
    - 定義⼀個 function 叫做 train_model，其中他會訓練模型以及紀錄 Accuracy / Loss 等資訊。
    - 透過 Matplotlib 等繪圖套件可以清楚視覺化訓練成果。
    - Dataset 使⽤ **Animal Image Dataset(DOG, CAT and PANDA)** 分成 train val 和 test set，其中訓練時只會使⽤到 train 和 val set，只有最終訓練完成測試時才會使⽤ test set。
    - Dataset Link: [DOG, CAT and PANDA](https://www.kaggle.com/datasets/ashishsaxena2209/animal-image-datasetdog-cat-and-panda)
    - 請把圖⽚⼤⼩設成 ==224 * 224==
    - 把所有的 seed 設為 0
3. 改進模型
    - 根據訓練成效，可以藉此修改模型或是訓練⽅式。例如，
        - 刪減或增多 CNN 或 FC 層數和⼤⼩、
        - 使⽤ learning rate scheduler 讓 learning rate 逐步遞減、
        - 使⽤augmentation來改變圖⽚顏⾊、翻轉圖⽚等。

