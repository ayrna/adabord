import numpy as np
from sklearn.base import clone
from malenia.metrics import amae
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score


class BaseAdaBoost(BaseEstimator):
    def __init__(
        self,
        estimator=None,
        n_estimators=10,
        learning_rate=0.5,
        softlabel=None,
        softlabel_alpha=0.1,
        penalization="linear",
        random_state=None,
        verbose=False,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.softlabel = softlabel
        self.softlabel_alpha = softlabel_alpha
        self.penalization = penalization
        self.models = []
        self.estimator_weights_ = []
        self.errors_ = []
        self.classes = []

        self._hist_w_range = []
        self._hist_error = []
        self._hist_alpha = []
        self._hist_test_mae = []
        self._hist_test_amae = []
        self.verbose = verbose

    def initialize_estimator(self):
        estimator = clone(self.estimator)
        if self.random_state is not None:
            if hasattr(estimator, "random_state"):
                estimator.random_state = self.random_state
            else:
                np.random.seed(self.random_state)
        return estimator

    def check_y(self, y):
        assert np.array_equal(np.unique(y), np.arange(len(np.unique(y))))

    def _update_histories(self, error, alpha, w, X_test, y_test):
        self._hist_w_range.append(np.max(w) - np.min(w))
        self._hist_error.append(error)
        self._hist_alpha.append(alpha)
        if X_test is not None and y_test is not None:
            self._hist_test_mae.append(mean_absolute_error(y_true=y_test, y_pred=self.predict(X_test)))
            self._hist_test_amae.append(amae(y_true=y_test, y_pred=self.predict(X_test)))

    def fit(self, X, y, X_test=None, y_test=None):
        self.check_y(y)
        self.classes = np.unique(y)
        w = np.ones(len(y)) / len(y)

        Q = len(self.classes)

        for step in range(self.n_estimators):
            model = self.initialize_estimator()
            # (a)
            model.fit(X, y, sample_weight=w)

            # (b)
            misclassified = model.predict(X) != y
            error = np.sum(w * misclassified) / np.sum(w)

            if error <= 0 + np.finfo(float).eps:
                print("Error is 0, stopping training")
                self.models.append(model)
                self.estimator_weights_.append(10.0)
                break
            else:
                # (c)
                alpha = np.log((1 - error) / error) + np.log(Q - 1)

                # (d)
                w = w * np.exp(alpha * misclassified)

                # (e)
                w /= np.sum(w)

                self.models.append(model)
                self.estimator_weights_.append(alpha)

                self._update_histories(error, alpha, w, X_test, y_test)

                if self.verbose:
                    self.print_progress(step, error, alpha, w, X_test, y_test)

        return self

    def print_progress(self, step, error, alpha, w, X_test, y_test):
        print(
            f"Step {step} ·· Error: {round(error, 4)} ·· Alpha: {round(alpha, 3)} ·· Range(w): {round(np.max(w) - np.min(w), 4)}"
        )
        print(
            f"Step {step} ·· Test MAE: {round(mean_absolute_error(y_true=y_test, y_pred=self.predict(X_test)), 4)}"
        )
        print(
            f"Step {step} ·· Test BAC: {round(balanced_accuracy_score(y_true=y_test, y_pred=self.predict(X_test)), 4)}"
        )
        print("")

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], len(self.classes)))
        for model, alpha in zip(self.models, self.estimator_weights_):
            proba += alpha * model.predict_proba(X)
        proba = softmax(proba, axis=1)
        assert np.allclose(np.sum(proba, axis=1), 1), "Probabilities do not sum to 1"
        return proba

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        pred = np.zeros((len(X), len(self.classes)))
        for m, model in enumerate(self.models):
            model_preds = model.predict(X)
            pred[np.arange(len(X)), model_preds] += self.estimator_weights_[m]

        return np.argmax(pred, axis=1)
