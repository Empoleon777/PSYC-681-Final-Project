import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def F1s(cols, true, pred):
    macrof1s = []
    microf1s = []

    for i in range(len(cols)):
        macro = f1_score(true[:, i], pred[:, i], average='macro')
        micro = f1_score(true[:, i], pred[:, i], average='micro')

        macrof1s.append(macro)
        microf1s.append(micro)

        print(f"Macro F1 ({cols[i]}): {macro}")
        print(f"Micro F1 ({cols[i]}): {micro}")
        print()

    print(f"Macro F1 (Average): {np.mean(macro)}")
    print(f"Micro F1 (Average): {np.mean(micro)}")

df = pd.read_csv(r"outputs/annotation_60k/gold_annotations.csv")

text = df['text']
q_cols = ['q01_relevance', 'q07_economic_direction', 'q08_social_direction']
y = df[q_cols].values

train_text, test_text, q_train, q_test= train_test_split(text, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9)
X_train = tfidf.fit_transform(train_text)
X_test = tfidf.transform(test_text)

clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
# clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf.fit(X_train, q_train)

preds = clf.predict(X_test)

# print(f"Macro F1: {f1_score(q_test, preds, average='macro')}")
# print(f"Micro F1: {f1_score(q_test, preds, average='micro')}")

F1s(q_cols, q_test, preds)