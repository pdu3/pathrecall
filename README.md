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
