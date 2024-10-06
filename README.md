# CSE 584 Midterm Project

**Jeongwon Bae (945397461)**
**Individual Participation**

## Overview

This project involves the development of an advanced deep learning classifier capable of identifying the specific Large Language Model (LLM) that generated a given text completion. The project encompasses dataset construction, classifier development, model training, and performance evaluation. By utilizing the DeBERTaV3 architecture, the classifier aims to elucidate differences between LLMs, with significant implications for model accountability and the detection of AI-generated content.

## Objectives

1. **Dataset Construction**: Curate a dataset from the BookCorpus, comprising truncated inputs (e.g., "Yesterday I went") and completions generated by seven distinct LLMs.
2. **Classifier Development**: Design a deep learning-based classifier leveraging DeBERTaV3 to accurately attribute a given text pair to its generating LLM.
3. **Performance Evaluation**: Assess classifier performance using metrics such as accuracy and micro-F1 score, complemented by a confusion matrix analysis.
4. **Related Work Analysis**: Situate the project within the context of prior research on LLMs, model attribution, and text classification.

## Dataset

The dataset for this project was constructed from the BookCorpus, consisting of 5,000 truncated text inputs generated by the following LLMs:

- Claude-3-haiku
- Falcon-7b
- Qwen2.5-3B-Instruct
- Phi-3-Mini-4K
- GPT-4o-mini
- Llama-2-7b-chat
- Llama-3.2-3B-Instruct

### Final Dataset Composition

| LLM Model             | Train  | Validation | Test  | Total  |
| --------------------- | ------ | ---------- | ----- | ------ |
| Claude-3-haiku        | 3,179  | 795        | 993   | 4,967  |
| Falcon-7b             | 3,168  | 792        | 990   | 4,950  |
| GPT-4o-mini           | 3,198  | 800        | 1,000 | 4,998  |
| Llama-2-7b-chat       | 3,182  | 796        | 994   | 4,972  |
| Llama-3.2-3B-Instruct | 3,198  | 799        | 1,000 | 4,997  |
| Phi-3-Mini-4K         | 3,197  | 800        | 1,000 | 4,997  |
| Qwen2.5-3B-Instruct   | 3,178  | 794        | 993   | 4,965  |
| **Total**             | 22,300 | 5,576      | 6,970 | 34,846 |

The completions were generated by providing each LLM with a standardized prompt, ensuring consistency across models for comparison purposes.

## Classifier Design

The classifier employs the DeBERTaV3 architecture, incorporating advanced features to enhance its performance.

### Models Used for Comparison

To establish a performance benchmark, several models were utilized for comparison:

- Multinomial Naive Bayes (MNB)
- K-Nearest Neighbors (KNN)
- Logistic Regression (LR)
- Random Forest (RF)
- Extreme Gradient Boosting (XGBoost)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Bidirectional LSTM (BiLSTM)
- Convolutional Neural Network with BiLSTM (CNNBiLSTM)
- BERT (Fine-tuned)
- XLNet (Fine-tuned)
- RoBERTa (Fine-tuned)

**DeBERTaV3Classifier**: The primary model, built on the DeBERTaV3 architecture, introduces disentangled attention mechanisms and an enhanced mask decoder, significantly improving pre-training efficiency and contextual representation. The model leverages multi-head attention and gradient-disentangled embedding sharing, making it adept at capturing subtle variations among different LLMs.

The architecture was specifically designed to distinguish between LLMs by effectively capturing the nuanced differences in their text generation patterns.

## Training and Evaluation

- The model was trained using the **AdamW optimizer** with a **linear learning rate scheduler** and warmup.
- **Hyperparameter tuning** was conducted via the Optuna library to identify optimal model settings.
- The training process spanned **20 epochs**, incorporating early stopping and gradient clipping to ensure stability and prevent overfitting.

## Results

| Model                               | ACC (↑)    | Micro-F1 (↑) |
| ----------------------------------- | ---------- | ------------ |
| Multinomial Naive Bayes (MNB)       | 0.3154     | 0.3062       |
| K-Nearest Neighbors (KNN) @k=7      | 0.1687     | 0.1646       |
| Logistic Regression (LR)            | 0.3976     | 0.3923       |
| Random Forest (RF)                  | 0.2634     | 0.2648       |
| Extreme Gradient Boosting (XGBoost) | 0.3581     | 0.3523       |
| RNN (Tokenizer-DeBERTaV3)           | 0.4258     | 0.4183       |
| LSTM (Tokenizer-DeBERTaV3)          | 0.4204     | 0.4162       |
| GRU (Tokenizer-DeBERTaV3)           | 0.4268     | 0.3983       |
| BiLSTM (Tokenizer-DeBERTaV3)        | 0.4446     | 0.4433       |
| CNNBiLSTM (Tokenizer-DeBERTaV3)     | 0.3914     | 0.3938       |
| BERT (Fine-tuned)                   | 0.5372     | 0.5369       |
| XLNet (Fine-tuned)                  | 0.5108     | 0.5139       |
| RoBERTa (Fine-tuned)                | 0.5202     | 0.5182       |
| **DeBERTaV3Classifier**             | **0.5478** | **0.5396**   |

The DeBERTaV3Classifier exhibited the highest performance compared to other baseline models, achieving an accuracy of **0.5478** and a micro-F1 score of **0.5396**. When the text generation length was extended to five sentences, the classifier's performance improved significantly, suggesting that additional context aids in distinguishing between LLMs more effectively.

## In-Depth Analysis and Additional Experiments

Further experimental evaluations were conducted to assess the robustness and effectiveness of the DeBERTaV3Classifier. Specifically, the classifier was tested on additional datasets to determine generalizability, and the impact of extending text generation from one to five sentences was explored.

### Applying DeBERTaV3Classifier to Other Datasets

- **BBC-text Multi-Class Dataset**: The classifier achieved an accuracy of **0.9752** and a micro-F1 score of **0.9753**, demonstrating its generalizability to standard text classification tasks.
- **Tweet Emotion Dataset**: On this dataset, the classifier achieved an accuracy of **0.9363** and a micro-F1 score of **0.9359**, indicating strong performance in emotion detection scenarios.

### Extending Generation of x_j to Five Sentences

The experiment involving the extension of text generation length to five sentences led to the exclusion of Llama-2-7b-chat and Falcon-7b due to inconsistencies in generated output. For the remaining five LLMs, the classifier's accuracy improved from **0.6074** to **0.8844**, while the micro-F1 score increased from **0.5954** to **0.8838**. This highlights that longer text samples provide richer contextual cues, which in turn enhance the classifier's capability to differentiate between LLMs.

### LLM Characteristic Analysis

The confusion matrices indicated varying degrees of success in distinguishing between LLMs. Analysis of bigrams and topics showed that common phrases like 'deep breath' were used by multiple models, while unique phrases like 'yesterday evening' helped differentiate specific models. Thematic analysis revealed that models had different focuses—some on abstract topics, others on temporal activities—enhancing classification, especially with longer text samples.

## Conclusion

This project underscores the feasibility of employing a deep learning-based approach for LLM attribution. The DeBERTaV3Classifier offers significant potential in areas such as AI-generated content detection and model accountability. Future research will aim to broaden the dataset by incorporating additional LLMs and further refine the classification methodology.

## How to Run the Code

To reproduce the results, follow the steps below. The main notebook (`main.ipynb`) is structured to be executed sequentially, allowing for the generation of the dataset and the training of the classifier by following the cells in order.

1. Clone the repository and navigate to the project folder.
   ```bash
   git clone [repository_link]
   cd CSE584_Midterm
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook to generate the dataset and train the classifier.
   ```bash
   jupyter notebook main.ipynb
   ```

## Repository Structure

- `CSE584_Midterm_JeongwonBae.pdf`: The project report detailing the methodology, experiments, and results.
- `main.ipynb`: Contains the implementation for dataset generation, classifier training, and evaluation.
- `README.md`: Overview and instructions for running the project.
- 
- `DataSet/`: Contains the datasets used for training and evaluation.
  - `X_i_samples_5k.csv`: Truncated text inputs used as prompts.
  - `X_jj_samples_5k_<ModelName>.csv`: Text completions generated by different LLMs.
  - `bbc-text.csv`, `tweet_emotions.csv`: Additional datasets used for evaluation.
- `DataSet_Generation/`: Scripts for generating the dataset.
  - `Construct_Xi.py`: Script to construct the truncated text inputs (Xi).
  - `LLM_<ModelName>.py`: Scripts for generating completions using different LLMs.
- `Model_Evaluation/`: Scripts for training and evaluating the models.
  - `DL_train_and_eval.py`: Script for deep learning model training and evaluation.
  - `Encoder_only_Classifier.py`, `base_model_ML.py`, `base_model_NN.py`: Scripts for different classifier architectures.
  - `load_and_preprocess.py`: Script to load and preprocess the datasets.

## References

Refer to the project report (`CSE584_Midterm_JeongwonBae.pdf`) for detailed information and a complete list of references.
