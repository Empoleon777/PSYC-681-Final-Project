import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.sparse import hstack

df = pd.read_csv("Data/MITweet.csv")

X_text = df.iloc[:, 1]
R_cols = [col for col in df.columns if col.startswith("R")]
I_cols = [col for col in df.columns if col.startswith("I")]
y_R = df[R_cols].values
y_I = df[I_cols].values

X_train, X_test, y_train_R, y_test_R, y_train_I, y_test_I = train_test_split(X_text, y_R, y_I, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

R_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
R_model.fit(X_train_tfidf, y_train_R)

R_train_feat = R_model.predict_proba(X_train_tfidf)
R_test_feat  = R_model.predict_proba(X_test_tfidf)

X_train_aug = hstack([X_train_tfidf, R_train_feat]).tocsr()
X_test_aug  = hstack([X_test_tfidf,  R_test_feat]).tocsr()

I_models = []
valid_masks = (y_train_I != -1)

for j in range(y_train_I.shape[1]):
    model = LogisticRegression(max_iter=1000)

    mask = valid_masks[:, j].astype(bool)

    y_j = y_train_I[mask, j]

    if len(np.unique(y_j)) < 2:
        continue

    model.fit(X_train_aug[mask], y_j)
    I_models.append(model)

I_pred = np.zeros_like(y_test_I)

for j, model in enumerate(I_models):
    I_pred[:, j] = model.predict(X_test_aug)

mask = (y_test_I != -1)

print("I Micro F1:", f1_score(
    y_test_I[mask],
    I_pred[mask],
    average='micro'
))

print("I Macro F1:", f1_score(
    y_test_I[mask],
    I_pred[mask],
    average='macro'
))