from abc import abstractmethod

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.class_weight import compute_class_weight


class MLPBaseClassifier(BaseEstimator, ClassifierMixin):
    @abstractmethod
    def __init__(self, learning_rate=1e-3, class_weight="balanced", max_iter=100):
        self.num_classes = None
        self.input_shape = None
        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.max_iter = max_iter

        self._class_weight = None

    def _initialize(self, X, y):
        self.classes_ = np.unique(y)
        self.num_classes = len(self.classes_)
        self.input_shape = X.shape[1]
        self.max_iter = int(self.max_iter)

        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        self._setup_model(self.num_classes, self.input_shape)

        if self.model is None:
            raise ValueError("No model was set up." + " Override the _setup_model method to define the model")

        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.75, patience=50, min_lr=1e-5, threshold=1e-2
        )

        if self.class_weight == "balanced":
            self._class_weight = compute_class_weight(class_weight="balanced", classes=self.classes_, y=y)
            self._class_weight = torch.tensor(self._class_weight, dtype=torch.float32)
        else:
            raise ValueError(f"Class weights {self.class_weight} not recognized")

        return self

    def _check_X_y(self, X, y):
        if X.shape[1] != self.input_shape:
            raise ValueError(
                f"Input shape {X.shape[1]} does not match expected" + " shape {self.input_shape}"
            )

        if len(set(y)) != self.num_classes:
            raise ValueError(
                f"Number of classes {len(set(y))} does not match expected number"
                + " of classes {self.num_classes}"
            )

        return X, y

    @abstractmethod
    def _setup_model(self, num_classes, input_shape):
        self.model = None
        return self

    @abstractmethod
    def _compute_loss(self, y, pred):
        return None

    def fit(self, X, y, verbose=False):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        first_pass = not hasattr(self, "model")
        if first_pass:
            self._initialize(X, y)

        self._check_X_y(X, y)

        if self.model is None:
            raise ValueError("No model was set up." + " Override the _setup_model method to define the model")

        device = next(self.model.parameters()).device

        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).to(device)

        self.model.train()
        for epoch in range(self.max_iter):
            pred = self.model(X)
            loss = self._compute_loss(y, pred)
            self.optimizer.zero_grad()
            if loss is None:
                raise ValueError("Loss is None. Override the _compute_loss method to define the loss")
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss)

            if verbose:
                print(f"Epoch {epoch+1}/{self.max_iter} - loss: {loss.item()}")

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("No model was set up." + " Override the _setup_model method to define the model")

        device = next(self.model.parameters()).device
        X = torch.from_numpy(X).float().to(device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probas = torch.softmax(logits, dim=1)
        return probas.cpu().numpy()

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
