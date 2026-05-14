#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re, string, joblib, requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. SCRAPING =================
urls = [
    "https://www.thehindu.com/news/",
    "https://www.ndtv.com/latest",
    "https://www.bbc.com/news",
    "https://www.reuters.com/",
    "https://indianexpress.com/",
    "https://timesofindia.indiatimes.com/",
    "https://www.cnn.com/",
    "https://www.aljazeera.com/",
    "https://www.usatoday.com/",
    "https://www.theguardian.com/international"
]

real_news = []
for url in urls:
    try:
        r = requests.get(url, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        for p in soup.find_all("p"):
            txt = p.get_text()
            if len(txt) > 50:
                real_news.append(txt)
    except:
        pass

real_df = pd.DataFrame(real_news[:15000], columns=["text"])
real_df["label"] = 1

# ================= 2. LOAD FAKE =================
fake_df = pd.read_csv("Fake.csv")
fake_df.columns = fake_df.columns.str.lower()
fake_df["text"] = fake_df["title"] + " " + fake_df["text"]
fake_df["label"] = 0
fake_df = fake_df[["text","label"]]

# ================= 3. CLEAN =================
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(f"[{string.punctuation}]", "", t)
    t = re.sub(r"\d+", "", t)
    return t

real_df["text"] = real_df["text"].apply(clean_text)
fake_df["text"] = fake_df["text"].apply(clean_text)

# ================= 4. COMBINE =================
df = pd.concat([real_df, fake_df], ignore_index=True)

# ================= 5. BALANCE =================
df_major = df[df.label == 1]
df_minor = df[df.label == 0]

df_minor_up = resample(df_minor, replace=True, n_samples=len(df_major), random_state=42)
df = pd.concat([df_major, df_minor_up]).sample(frac=1).reset_index(drop=True)

# ================= 6. STYLOMETRIC FEATURES =================
def caps_ratio(text):
    words = text.split()
    return sum(1 for w in words if w.isupper()) / (len(words)+1)

def exclamation_density(text):
    return text.count("!") / (len(text)+1)

df["caps"] = df["text"].apply(caps_ratio)
df["exclaim"] = df["text"].apply(exclamation_density)

# ================= 7. TF-IDF =================
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
X_text = tfidf.fit_transform(df["text"])

# Combine features
X_extra = df[["caps","exclaim"]].values
X = np.hstack((X_text.toarray(), X_extra))
y = df["label"]

# ================= 8. SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ================= 9. MODELS =================
models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(kernel="rbf", C=10)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"\n{name} Accuracy:", acc)
    print(classification_report(y_test, pred))

# ================= 10. GRAPH =================
plt.bar(results.keys(), results.values())
plt.title("Model Comparison")
plt.show()

# ================= 11. SAVE =================
joblib.dump(models["SVM"], "svm_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")

# ================= 12. BERT =================
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

dataset = Dataset.from_pandas(df[["text","label"]])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(x):
    return tokenizer(x["text"], padding="max_length", truncation=True)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.train_test_split(test_size=0.2)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

args = TrainingArguments(
    output_dir="./bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()

model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")


# In[2]:


preds = trainer.predict(dataset["test"])
print(preds.metrics)


# In[3]:


trainer.predict(dataset["test"])


# In[4]:


from transformers import pipeline

clf = pipeline("text-classification", model="./bert_model")
print(clf("US is attacking iran"))


# In[5]:


svm_model = joblib.load("svm_model.pkl")
tfidf = joblib.load("tfidf.pkl")


# In[6]:


from transformers import pipeline

clf = pipeline("text-classification", model="./bert_model")

print(clf("US is attacking iran"))


# In[7]:


result = clf("US is attacking india")[0]

label = result['label']

if label == "LABEL_1":
    print("Real News")
else:
    print("Fake News")


# In[ ]:





# In[ ]:




