#SMS Spam Detector

A practical machine learning project to classify SMS messages as Spam or Ham (Not Spam) using Natural Language Processing and classic classification models.

##Overview

This notebook demonstrates a step-by-step pipeline to clean SMS text messages, transform them into numerical representations, and train predictive models to detect spam. The approach is lightweight, interpretable, and effective for real-world spam filtering tasks.

##What’s Inside

📄 `SMS_spam_classifier.ipynb`
➡Jupyter Notebook with complete workflow: data loading → preprocessing → model training → evaluation.

###Main Highlights:

* Text cleaning & preprocessing (punctuation removal, stemming, stopword filtering)
* Feature extraction using **TF-IDF Vectorizer**
* Trained models:
  * Multinomial Naive Bayes
  * Support Vector Machines (SVM)
  * Random Forest Classifier
* Evaluation via confusion matrix & metrics
* Model serialization using `pickle` for reuse


##Installation & Setup

Before running the notebook, make sure to install all required libraries:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

You also need to download some NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Preprocessing Pipeline

* Lowercasing text
* Removing punctuation and special characters
* Tokenization
* Stopword removal
* Stemming using PorterStemmer
* TF-IDF vectorization

---

##Model Usage Example

After training and saving the model, you can use it like this:

```python
predict("Claim your free cash prize now!")
# Output: 'spam'
```

---

##Evaluation Metrics

* Accuracy Score
* Precision & Recall
* Confusion Matrix
* Classification Report

> Typically achieves \~98% accuracy using Naive Bayes

---

##Dataset Details

* Source: `spam.csv`
* Fields:

  * `label`: `ham` or `spam`
  * `text`: SMS message content

