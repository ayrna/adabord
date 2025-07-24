import numpy as np
from adabord_base import BaseAdaBoost


def norm_rank_probability_score(y, y_predprob, penalization):
    y = np.array(y)
    n_classes = y_predprob.shape[1]

    if not y.shape == y_predprob.shape:
        if not np.allclose(np.unique(y), np.arange(n_classes)):
            raise ValueError(
                f"Label error: y unique is {np.unique(y)} and y_predprob shape is {y_predprob.shape}"
            )
        y_oh = np.zeros(y_predprob.shape)
        y_oh[np.arange(len(y)), y] = 1
        y = y_oh

    y = y.cumsum(axis=1)
    y_predprob = y_predprob.cumsum(axis=1)

    if penalization == "linear":
        return np.sum(np.abs(y_predprob - y), axis=1) / (n_classes - 1)
    elif penalization == "quadratic":
        return np.sum(np.power(y_predprob - y, 2), axis=1) / (n_classes - 1)
    else:
        raise ValueError(f"Unknown penalization: {penalization}. Use 'linear' or 'quadratic'.")


class ADABORDClassifier(BaseAdaBoost):

    def fit(self, X, y, X_test=None, y_test=None):
        self.check_y(y)
        self.classes = np.unique(y)
        w = np.ones(len(y)) / len(y)

        for step in range(self.n_estimators):
            model = self.initialize_estimator()
            # (a)
            model.fit(X, y, sample_weight=w)

            # (b)
            if self.softlabel is not None:
                y_soft = self._softlabel(y)
                error_per_sample = norm_rank_probability_score(
                    y_soft, y_predprob=model.predict_proba(X), penalization=self.penalization
                )
            else:
                error_per_sample = norm_rank_probability_score(
                    y, y_predprob=model.predict_proba(X), penalization=self.penalization
                )
            error = np.sum(w * error_per_sample) / np.sum(w)
            self.errors_.append(error)

            if error <= 0 + np.finfo(float).eps:
                print("Error is 0, stopping training")
                self.models.append(model)
                self.estimator_weights_.append(10.0)
                break
            # elif error >= 1 - np.finfo(float).eps:
            #     raise ValueError("Error is 1, ensemble cannot be fitted")
            else:
                # (c)
                alpha = np.log((1 - error) / error)

                # (d)
                misclassified = model.predict(X) != y
                w = w * np.exp(alpha * misclassified)

                # (e)
                w /= np.sum(w)

                self._update_histories(error, alpha, w, X_test, y_test)

                if self.verbose:
                    self.print_progress(step, error, alpha, w, X_test, y_test)

                self.models.append(model)
                self.estimator_weights_.append(alpha)
