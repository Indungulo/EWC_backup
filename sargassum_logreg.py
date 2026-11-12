import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

db = pd.read_csv('/mnt/user-data/outputs/sargassum_feature_db.csv')

# Oa09, Oa12, Oa16 radiances plus the two indices
feature_cols = ['L_673.75', 'L_753.75', 'L_778.75', 'SGI', 'AFAI']
X = db[feature_cols].values
y = db['is_sargassum'].values

idx_train, idx_test = train_test_split(np.arange(len(db)), test_size = 0.3, stratify = y, random_state = 42)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter = 1000))
model.fit(X[idx_train], y[idx_train])
y_pred = model.predict(X[idx_test])

print(confusion_matrix(y[idx_test], y_pred))
print(classification_report(y[idx_test], y_pred, target_names = ['other', 'sargassum'], digits = 3))

# Same test set, broken down by scene
test = db.iloc[idx_test].assign(pred = y_pred)
for product, grp in test.groupby('product'):
    print(product[16:31], len(grp), 'errors:', (grp['pred'] != grp['is_sargassum']).sum())

# Misclassified test pixels
print(test[test['pred'] != test['is_sargassum']][['product', 'Name', 'label', 'SGI', 'AFAI']])

coef = pd.Series(model[-1].coef_[0], index = feature_cols).sort_values(key = np.abs, ascending = False)
print(coef.round(3))
