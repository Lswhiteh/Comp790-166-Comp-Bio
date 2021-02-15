import community as louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmis
from sklearn.metrics import plot_roc_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

edges = np.genfromtxt("../data/cell_graph.edgelist")
levmat = pd.read_csv("../data/Levine_matrix.csv", header=0)
X = levmat.dropna()

# 3-1
default_vec = np.genfromtxt("../embs/default.emb", skip_header=1)
ids = default_vec[:, 0]
vecs = default_vec[:, 1:]

clusts = KMeans(24).fit_predict(vecs)

trueparts = []
predparts = []
for id, pred in zip(ids, clusts):
    if not pd.isnull(levmat.loc[int(id), "label"]):
        trueparts.append(int(levmat.loc[int(id), "label"]))
        predparts.append(int(pred))

print("NMIS for 24 K-means clustering for 128 dim. vector:", nmis(trueparts, predparts))
print("--------------------------------------------------------", "\n")

# It does just slightly better than clustering on the raw marker data, but worse than the graph partitioning

# 3-2
dimlist = ["48", "68", "128", "168", "200"]
nmis_scores = []
for dims in dimlist:
    vec = np.genfromtxt("../embs/{}.emb".format(dims), skip_header=1)
    ids = vec[:, 0]
    vecs = vec[:, 1:]

    clusts = KMeans(24).fit_predict(vecs)

    trueparts = []
    predparts = []
    for id, pred in zip(ids, clusts):
        if not pd.isnull(levmat.loc[int(id), "label"]):
            trueparts.append(int(levmat.loc[int(id), "label"]))
            predparts.append(int(pred))

    score = nmis(trueparts, predparts)
    nmis_scores.append(score)
    print("NMIS for 24 K-means clustering for {} dim. vector:".format(dims), score)
print("--------------------------------------------------------", "\n")

# It seems like there's not much of a difference, but the 68 and 168 dimension vectors have the best on average, usually they're equivalent to 128 though.

intdims = [int(i) for i in dimlist]
plt.plot(intdims, nmis_scores)
plt.xlabel("Dimensions")
plt.ylabel("Score")
plt.title("Dims vs Score")
plt.savefig("Dims_vs_Score.png")

# 3-3
plist = ["0.25", "0.5", "1", "3", "10"]
nmis_scores = []
for p in plist:
    vec = np.genfromtxt("../embs/{}.emb".format(p), skip_header=1)
    ids = vec[:, 0]
    vecs = vec[:, 1:]

    clusts = KMeans(24).fit_predict(vecs)

    trueparts = []
    predparts = []
    for id, pred in zip(ids, clusts):
        if not pd.isnull(levmat.loc[int(id), "label"]):
            trueparts.append(int(levmat.loc[int(id), "label"]))
            predparts.append(int(pred))

    score = nmis(trueparts, predparts)
    nmis_scores.append(score)
    print("NMIS for 24 K-means clustering for p of {} :".format(p), score)

# Again not noticing much of a difference, wonder if I need to just go crazy with the parameterizations?

fltp = [float(i) for i in dimlist]
plt.clf()
plt.plot(fltp, nmis_scores)
plt.xlabel("P Parameter")
plt.ylabel("Score")
plt.title("P vs Score")
plt.savefig("P_vs_Score.png")
print("--------------------------------------------------------", "\n")


# 3-4
vec_df = pd.read_csv("../embs/128.emb", header=None, index_col=0, sep=" ")
combo = X.join(vec_df, how="inner")

tcells = combo.loc[combo["label"].isin([11, 12, 17, 18]), :]
mcells = combo.loc[combo["label"].isin([1, 2, 3]), :]

tcellsmat = np.array(tcells.loc[:, "CD45":"CD3"])
mcellsmat = np.array(mcells.loc[:, "CD45":"CD3"])

tlab = np.zeros(len(tcellsmat))
mlab = np.ones(len(mcellsmat))

both_mat = np.vstack([tcellsmat, mcellsmat])
both_lab = np.hstack([tlab, mlab])

X_train, X_test, y_train, y_test = train_test_split(
    both_mat, both_lab, shuffle=True, stratify=both_lab, test_size=0.33
)
# print(X_train, X_test, y_train, y_test)
clf = make_pipeline(StandardScaler(), LinearSVC(random_state=42))
clf.fit(X_train, y_train)
preds = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, preds)
print("FPR:", fpr)
print("TPR:", tpr)
plot_roc_curve(clf, X_test, y_test)
plt.savefig("ROC_3-4.png")

# ... Why is this also perfect
