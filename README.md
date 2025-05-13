
# **Email Spam Detection**

This project implements a Machine Learning model to detect spam emails using a dataset **(spam2.csv)**. The goal is to accurately classify emails as **"Spam"** or **"Not Spam" (ham)**. The model leverages various **NLP** preprocessing techniques and uses a **deep learning architecture for classification**.




## **Table of Contents**

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture and Training](#model-architecture-and-training)
- [Results](#results)
- [Example](#example)
- [Acknowledgements](#acknowledgements)
## **Overview**

The project uses a labeled email dataset where each record consists of the email label (either **ham** for **non-spam** or **spam**), the email message, and some extra columns that are removed during preprocessing. The steps covered in this project include data loading, balancing, text cleaning (removing punctuations and stop words), visualization (word clouds), tokenization, padding, model training, and prediction.
## **Dataset**

The dataset used is **spam2.csv** which has the following format:

| v1   | v2                                                       | Unnamed: 2 | Unnamed: 3 | Unnamed: 4 |
|------|----------------------------------------------------------|------------|------------|------------|
| ham  | Go until jurong point, crazy.. Available only ...        | NaN        | NaN        | NaN        |
| ham  | Ok lar... Joking wif u oni...                            | NaN        | NaN        | NaN        |
| spam | Free entry in 2 a wkly comp to win FA Cup fina...        | NaN        | NaN        | NaN        |
| ham  | U dun say so early hor... U c already then say...        | NaN        | NaN        | NaN        |
| ham  | Nah I don't think he goes to usf, he lives aro...        | NaN        | NaN        | NaN        |

## **Technologies Used**

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- NLTK (stopwords)
- TensorFlow / Keras
- Scikit-learn
- WordCloud
- Pickle


## **Project Structure**

```bash
  Email-Spam-Detection/
├── Spam Detection.ipynb          # Main Jupyter notebook with the complete project code
├── spam_model.keras              # Trained LSTM model saved in Keras format
├── spam2.csv                     # Dataset containing spam and ham email data
├── tokenizer.pkl                 # Tokenizer used for text preprocessing
├── requirements.txt              # Python dependencies needed to run the project
```
## **Installation**

1) **Clone the repository:**

```bash
  git clone https://github.com/Vidhi1155/spam-email-detection.git
```

2) **Navigate to the project directory:**

```bash
  cd Email-Spam-Detection
```

3) **Create a virtual environment (optional but recommended):**

```bash
  python -m venv venv
  source venv/bin/activate      # On Windows: venv\Scripts\activate
```

4) **Install the required dependencies using requirements.txt:**

```bash
  pip install -r requirements.txt
```
5) **Launch the Jupyter Notebook:**

```bash
  jupyter notebook
```
Open the **Spam Detection.ipynb** file to run the project.



## **Usage**

Run the main file **Spam Detection.ipynb** to:

* Load and preprocess the spam2.csv dataset.
* Balance the dataset between "ham" and "spam" emails.
* Clean and process the email text by removing punctuations and stop words.
* Visualize data distribution and generate word clouds.
* Tokenize and pad the email sequences.
* Build and train an LSTM-based deep learning model.
* Evaluate the model on a test set.
* Save the trained model and tokenizer.
* Predict the class of new email examples.

For example, to run the prediction on a sample email, simply call the provided **predict_email function** with the desired text.


## **Model Architecture and Training**
The deep learning model is defined using **TensorFlow** and **Keras**:

* **Embedding Layer**: Converts tokenized text into dense vectors.
* **Bidirectional LSTM**: Captures temporal dependencies from both directions.
* **Dense Layers**: Process and classify the features.
* **Dropout Layer**: Helps prevent overfitting.
* **Optimizer**: Adam with an Exponential Decay learning rate schedule.
* **Callbacks**: EarlyStopping and ReduceLROnPlateau are used to optimize training performance.

The model is compiled with **binary crossentropy loss** and **accuracy** as the **evaluation metric**.
## **Results**

During training, the model achieved a final training **accuracy** of approximately **97.74%** on the available test set. Various test examples indicate that the model can effectively classify whether an email is spam or not.
## Example

1) #### ✅ **Example 1: Spam (Prediction: 1.00)**
Dear Customer, We have detected unusual activity on your account. To restore access, please verify your details immediately: [Click Here to Verify] Failure to do so will result in permanent suspension.

2) #### ✅ **Example 2: Spam (Prediction: 0.71)**
Hello Michael, Your order of "Noise Cancelling Headphones" has been shipped and will arrive by Tuesday. You can track your package here: [Track Package] Thank you for shopping with us.

3) #### ❌ **Example 3: Not Spam (Prediction: 0.36)**
Hi John, Just a quick reminder that we have our project update meeting scheduled for 10 AM tomorrow in Conference Room B. Please bring the latest sprint report.

4) #### ❌ **Example 4: Not Spam (Prediction: 0.31)**
This is a confirmation that you paid ₹499 for your monthly Netflix subscription on May 5th, 2025. You can view your invoice here.

## **Acknowledgements**

- [Flickr8k Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — Source of the spam email data (`spam2.csv`)
- [NLTK](https://www.nltk.org/) — For natural language processing tools like stopwords
- [TensorFlow & Keras](https://www.tensorflow.org/) — For building and training the LSTM model
- [Scikit-learn](https://scikit-learn.org/) — For train-test splitting and ML utilities
- [Matplotlib & Seaborn](https://matplotlib.org/) — For visualizations and insights
- [WordCloud](https://github.com/amueller/word_cloud) — For generating word cloud visualizations


