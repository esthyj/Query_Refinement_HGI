# ✨ Query Refinement
**Position:** Intern at Hanwha General Insurance  
**Project Duration:** 2025.06.16 - 2025.06.26  
**Project Title:** User Query Refinement  
**[Task Description]**  
1. Detection of ambiguity in user queries  
2. Refinement and clarification of ambiguous queries  
3. Classification of user query intent  

## 📂 Project Directory Structure
📦 Query_Refinement_HGI/  
├─ 📂 LLM_Query_Refinement/                     # 3Task_LLM
│ ├─ 📜 dataset_3labels.csv                    # Dataset of Support/Division/Irrelevant  
│ ├─ 📜 query_refinement.py                    # Doing 3 tasks simultaneously by llm   
│ ├─ 📜 query_refinement_1try.csv              # dataset_3labels.csv file through query_refinement.py  
│ └─ 📜 query_refinement_2try.csv             # query_refinement_2try.csv file through query_refinement.py   
├─ 📂 NL_Classification/                        # Doing query intent classification by classification model  
│ ├─ 📜 classifier_model.ipynb                 # Doing query intent classification by classification model  
│ ├─ 📜 dataset_3labels_class_train.csv        # Train File  
│ └─ 📜 query_refinement_2try.csv             # Test File  
└─ 📜 README.md                                # Project overview  


flowchart LR
    A[Hard] -->|Text| B(Round)
    B --> C{Decision}
    C -->|One| D[Result 1]
    C -->|Two| E[Result 2]
    
## 🔎 Results
||Original RAG Pipeline|3Task_LLM|NL Classification
|---|---|---|---|
|model|dnotitia/Smoothie-Qwen3-14B|dnotitia/Smoothie-Qwen3-14B(8bit)|klue/bert-base
|Ambiguity Detection Accuracy(↑)|72/90 (Acc: 0.8)|-|-|
|Clarification Suggestion Retry Rate(↓)|0.07|0||
|Intent Classification Accuracy(↑)|[2 labels]<br>59/60 (Acc:0.98)|[3 labels]<br>74/90 (Acc:0.82)|[3 labels]<br>16/18 (Acc:0.89)|
|Average Response Time(↓)|Clear:0.9s<br>Ambiguous:2.5s|Clear:2.9s<br>Ambiguous:8.5s|0.001s(very fast)|
|Remarks||Variations in the environment may result in slower average response times||
