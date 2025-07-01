# 🎉 GPTVD
This repository contains the code and data for our GPTVD project. 🚀
## 📂 Repository Structure
```
project-root/
├── gptvd/
│ ├── clusters.py
│ ├── llm.py
│ ├── cluster_ablation_data_split.py
│ └── eval.py
├── data/
└── README.md
```
### 🛠️ `gptvd/`
Contains all of the Python code used to run and analyze the GPTVD:
- **clusters.py**  
  Script to run clustering on the extracted representations.📊
- **llm.py**  
  Utilities for querying and interacting with LLM.🤖
- **cluster_ablation_data_split.py**  
  Code for generating ablation splits of the data for cluster analysis.🔍
- **eval.py**  
  Evaluation scripts.✅
### `data/`
Holds the handcrafted prompt samples and the pre-trained clustering models:
- **`*.json` files**  
  - `val_part.json`: some test results.
  - `representatives_label*.json`: the representative examples selected for each cluster label.
- **`*.pkl` files**  
  - `kmeans_label*.pkl`: pre-trained K-means models for each setting of the number of clusters.
## ☁️ Downloading the Data
The full dataset (JSON files and pretrained models) is hosted on Google Drive. To download:
[![Google Drive Folder](https://img.shields.io/badge/Google%20Drive-%20Download-blue)](https://drive.google.com/drive/folders/1dIhwdxqVRbJLr2O1eLfj5-OW4G1e9UAy?usp=sharing)

> **🔗 Link:**  
> https://drive.google.com/drive/folders/1dIhwdxqVRbJLr2O1eLfj5-OW4G1e9UAy?usp=sharing

Simply click the badge or the link above, then use “Add to My Drive” or “Download” to grab the files.
