# src/pipeline/pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from src.feature_selection.hybrid import HierarchicalPCA

def build_hybrid_pipeline(selected_features, n_clusters_options=[10,15,20]):
    pipeline = Pipeline([
        ('feature_selection', ColumnTransformer([
            ('selector', 'passthrough', selected_features)
        ])),
        ('hpca', HierarchicalPCA(n_clusters=15)),
        ('classifier', LogisticRegression(solver='saga', max_iter=1000))
    ])
    
    param_grid = {
        'hpca__n_clusters': n_clusters_options,
        'classifier': [
            DecisionTreeClassifier(),
            GaussianNB(),
            LogisticRegression(max_iter=1000),
            KNeighborsClassifier(n_neighbors=5),
            SVC(kernel='rbf', C=1),
            MLPClassifier(max_iter=1000)
        ]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
    return grid_search
