from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
import sklearn
from skada.datasets import make_shifted_datasets
from skada.metrics import PredictionEntropyScorer, CircularValidation
from sklearn.model_selection import cross_val_score
from skada.model_selection import SourceTargetShuffleSplit

X, y, sample_domain = make_shifted_datasets(
    20, 20, shift="conditional_shift", random_state=42
)
# Initial configuration
clf = SVC(probability=True)
scorer = PredictionEntropyScorer()
cv = SourceTargetShuffleSplit(random_state=0)

scores = cross_valscores = cross_val_score(
    clf, X, y, params={"sample_domain": sample_domain}, cv=cv, scoring=scorer
)
print(f"Entropy score: {scores.mean():1.2f} (+-{scores.std():1.2f})")

# Check if the scorer is at fault
clf2 = SVC(probability=True)
scorer2 = CircularValidation()
scores2 = cross_val_score(
    clf, X, y, params={"sample_domain": sample_domain}, cv=cv, scoring=scorer2
)
# It's not.

# Check if the skada scorers is at fault
clf2 = SVC(probability=True)
scorer3 = sklearn.metrics.f1_score
scorer3 = sklearn.metrics.make_scorer(scorer3, average="macro")
scores3 = cross_val_score(
    clf, X, y, params={"sample_domain": sample_domain}, cv=cv, scoring=scorer3
)
# They are : no future warning with sklearn.metrics.f1_score