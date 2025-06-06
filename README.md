# Deep Learning Approach for Analyzing Sentiment in Tamil Social Media Posts

## Overview
This project focuses on political multiclass sentiment analysis of Tamil social media posts, particularly from the platform X (formerly Twitter). The goal is to classify sentiments into seven categories: substantiated, sarcastic, opinionated, positive, negative, neutral, and none of the above.

## Dataset
- **Source**: DravidianLangTech@NAACL 2025 shared task.
- **Size**: 3,352 training samples and 544 validation samples.

| Label             | Training Samples | Validation Samples |
|----------------|-----------------|--------------------|
| Opinionated   | 1,361                     | 153                           |
| Sarcastic         | 790                        | 115                           |
| Neutral            | 637                        | 84                             |
| Positive            | 575                        | 69                             |
| Substantiated | 412                        | 52                             |
| Negative          | 406                        | 51                             |
| None of the Above | 171                 | 20                             |

## Preprocessing
- **Handling Class Imbalance**: Synthetic Minority Over-sampling Technique (SMOTE) is applied.
- **Text Normalization and Tokenization**: Using IndicNLP library.
- **Vectorization**: Count Vectorizer for traditional models.

## Models Implemented
1. **Complement Naive Bayes (CNB)**: 
   - Adjusts weights using the complement of each class to handle imbalanced datasets.

2. **Voting Classifier** (Ensemble Model):
   - Combines Decision Tree, SVM, Naive Bayes, K-Nearest Neighbors, and Logistic Regression using soft voting.

3. **Deep Learning Model (LSTM)**:
   - Two stacked LSTM layers with 64 and 32 units, dropout layers, and softmax output.

4. **Transfer Learning Model (DeBERTa V3)**:
   - Utilizes disentangled attention mechanism and absolute position embeddings for better contextual understanding.

5. **Hybrid Model (DeBERTa + LSTM)**:
   - Combines DeBERTa embeddings with LSTM layers for sequential dependency capture.

## Results
| Model                    | Accuracy | Precision | Recall | Macro-F1 Score |
|----------------|-----------|------------|---------|----------------|
| Naive Bayes      | 0.3382      | 0.3367         | 0.2962   | 0.3059                |
| Voting Classifier | 0.3750      | 0.3387         | 0.3250   | 0.3227                |
| LSTM                     | 0.2868      | 0.3252         | 0.2803   | 0.2964                |
| DeBERTa V3          | 0.2610      | 0.0761         | 0.1499   | 0.0986                |
| Hybrid (DeBERTa + LSTM) | 0.3162 | 0.3143 | 0.2997 | 0.3026 |

This work secured the **10th** position in the **DravidianLangTech@NAACL 2025 shared task on Political Sentiment Analysis in Tamil Social Media Posts**, highlighting the effectiveness of our hybrid and ensemble-based approaches. [Link to results](https://codalab.lisn.upsaclay.fr/competitions/20702#learn_the_details-results)

## Conclusion
- The **Voting Classifier** achieved the best performance due to ensemble learning.
- **DeBERTa** struggled due to the complexity of the dataset.
- The **Hybrid Model** leveraged the strengths of both contextual embeddings and sequential learning.

## Limitations and Future Work
- Address class imbalance more effectively.
- Explore multimodal approaches.
- Optimize DeBERTa’s computational cost for real-time deployment.
- Extend the dataset and evaluate performance across different Tamil dialects.

## Research Work
[@inproceedings{abhay2025MNLP, title={MNLP@DravidianLangTech 2025: Transformers vs. Traditional Machine Learning: Analyzing Sentiment in Tamil Social Media Posts}, author={Vishwakarma, Abhay and Kumar, Abhinav}, booktitle={Proceedings of Fifth Workshop on Speech and Language Technologies for Dravidian Languages (NAACL 2025)}, pages={}, year={2025} }](https://aclanthology.org/2025.dravidianlangtech-1.72/)
