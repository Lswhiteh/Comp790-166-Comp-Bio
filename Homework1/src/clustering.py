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


def plot_coefficients(classifier, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    plt.xticks(np.arange(1, 1 + 2 * top_features), rotation=60, ha="right")
    plt.savefig("coeff.png")


G = np.genfromtxt("../data/cell_graph.edgelist", delimiter=" ")
popnames = pd.read_csv("../data/population_names_Levine_13dim.txt", header=0, sep="\t")
levmat = pd.read_csv("../data/Levine_matrix.csv", header=0)
X = levmat.dropna()


# 2-1
clusts = KMeans(24).fit_predict(np.array(X.loc[:, "CD45":"CD3"]))
print("NMIS for 24 K-means clustering:", nmis(np.array(X.loc[:, "label"]), clusts))

# 2-2
Gx = nx.from_edgelist(G)
predparts = []
trueparts = []
for node, part in louvain.best_partition(Gx).items():
    if not pd.isnull(levmat.loc[int(node), "label"]):
        trueparts.append(int(levmat.loc[int(node), "label"]))
        predparts.append(int(part))

print("NMIS for Louvain partitioning:", nmis(trueparts, predparts))

# 2-3
# The graph partitioning has a better NMIS score by a decent margin, so I would choose that over clustering on X. I will say, however, that doing clustering on X without the limitation of keeping 24 clusters you can achieve a higher NMIS than either (around 86 for k=11), which makes me wonder how NMIS could be misused.

# 2-4
clusts_21_x = []
for i, j in zip(X.loc[:, "label"].tolist(), clusts):
    if int(i) == 21:
        clusts_21_x.append(j)
print("Number of unique clusters where pDCs present in X:", len(set(clusts_21_x)))
print("Clusters:", clusts_21_x)
clusts_21_g = []
for i, j in zip(trueparts, predparts):
    if i == 21:
        clusts_21_g.append(j)

print("Number of unique clusters where pDCs present in G:", len(set(clusts_21_g)))
print("Clusters:", clusts_21_g)


# 2-5
tcells = X.loc[X["label"].isin([11, 12, 17, 18]), :]
mcells = X.loc[X["label"].isin([1, 2, 3]), :]

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
plt.savefig("ROC_2-5.png")
# Why is it perfect?

print(X.columns[:-1])
print(clf.steps[1][1].coef_)
