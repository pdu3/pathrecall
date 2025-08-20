# PathRecall: Retrieval-Augmented Visual Question Answering for Indoor Navigation

## 📝 Overview
**PathRecall** is a research and engineering prototype that explores how *retrieval-augmented temporal memory* can enhance **Visual Question Answering (VQA)** in **indoor navigation** scenarios.  
Unlike conventional VQA models that operate on a single image, PathRecall maintains an evolving memory of previously encountered frames.  
This enables answering context-dependent queries such as:  
> *“Earlier, when I passed by the recycling area with the number 3 sign, what kind of seating did I encounter?”*

The system integrates a custom **MemoryNet retriever**, a modified **BLIP-2 backbone**, and a **RAG-style pipeline**, showing how temporal memory retrieval improves accessibility of built environments for **visually impaired (VI)** users.

---

## 🔑 Key Features
- **Temporal Memory Retrieval**: Maintains an index of past video frames for history-aware QA.  
- **MemoryNet**: A lightweight retriever trained with MiniLM embeddings + BLIP-2 features.  
- **Question-Aware Captioning**: Generates scene descriptions conditioned on the query.  
- **RAG Pipeline**: Connects retrieval with answer generation to ensure grounded responses.  
- **Assistive Focus**: Designed with applications in accessibility and indoor navigation in mind.  

---

## 📂 Project Structure
Pathrecall/                 # 这是你的项目根目录
├── src/                    # "source"，源码文件夹
│   ├── memorynet/          # 存放 MemoryNet 的代码（训练 & 推理）
│   ├── blip2_mod/          # 存放你修改过的 BLIP-2 模块
│   ├── rag_pipeline/       # 存放 RAG（检索+生成）的代码
│
├── data/                   # 数据目录（太大，不上传 GitHub）
│   ├── features/           # 存 BLIP-2 提取的特征（.npy等）
│   ├── memoryrank.jsonl    # MemoryNet 训练数据
│   └── ...                 
│
├── notebooks/              # 存放 Jupyter Notebook，用来做实验 & 可视化
│   ├── train_memorynet.ipynb
│   ├── evaluation.ipynb
│   └── ...
│
├── results/                # 存放结果，比如模型输出、图表、论文里用的图
│   ├── figures/            # 绘图（精度曲线、示意图等）
│   ├── samples/            # 示例结果（输入问题 + 检索帧 + 输出答案）
│   └── evaluation.md       # 实验报告（详细的评价指标）
│
└── README.md               # 项目说明文档


---

## ⚙️ Installation
```bash
git clone https://github.com/pdu3/pathrecall.git
cd pathrecall
conda create -n pathrecall python=3.10
conda activate pathrecall
pip install -r requirements.txt

## 🖼️ Example Usage
from pathrecall import MemoryNet, RAGPipeline

retriever = MemoryNet.load_pretrained("checkpoints/memorynet.pt")
pipeline = RAGPipeline(retriever=retriever)

question = "When I looked up at the green exit sign, which direction should I have gone?"
answer, frames = pipeline.answer(question)

print("Answer:", answer)

