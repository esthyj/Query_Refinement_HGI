# ✨ Query Refinement
**Position:** Intern at Hanwha General Insurance  
**Project Duration:** 2025.06.16 - 2025.06.26  
**Project Title:** User Query Refinement  
**[Task Description]**  
1. Detection of ambiguity in user queries  
2. Refinement and clarification of ambiguous queries  
3. Classification of user query intent  

## 📂 Project Directory Structure
Query_Refinement_HGI/
├──LLM_Query_Refinement/                     # Doing 3 tasks simultaneously by llm
│ ├── dataset_3labels.csv                    # Dataset of Support/Division/Irrelevant
│ ├── query_refinement.py                    # Llm prompt
│ ├── query_refinement_1try.csv              # dataset_3labels.csv file through query_refinement.py 
│ └──  query_refinement_2try.csv             # query_refinement_2try.csv file through query_refinement.py 
├──NL_Classification/                        # Doing query intent classification by classification model
│ ├── classifier_model.ipynb                 # Doing query intent classification by classification model
│ ├── dataset_3labels_class_train.csv        # Train File
│ └──  query_refinement_2try.csv             # Test File
└── README.md                                # Project overview
