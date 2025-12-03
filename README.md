# border-collie-classifier
AIoT-HW4 with demo02
邊境牧羊犬辨識器（AIoT 第四次作業）
本專案參考【Demo02】遷移式學習做八哥辨識器.ipynb，將任務改為「邊境牧羊犬 vs 其他狗」，並加入資料重組、下採樣、資料增強與雲端部署。

主要更動

資料：Kaggle 120犬種，抽出 n02106166-Border_collie，其餘合併為 other_dog；控制比例約 150:450。

模型：MobileNetV2 遷移學習→凍結訓練→部分解凍微調；改用 Keras 新格式儲存以利部署。

部署：Streamlit 應用，上傳圖片即時推論。

快速開始

pip install -r requirements.txt

streamlit run app/streamlit_app.py
