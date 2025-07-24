from mlp_base import MLPBaseClassifier
from torch.nn import CrossEntropyLoss
from torch import nn
import torch


class MLPModel(nn.Module):
    def __init__(self, input_shape, num_classes, hidden_neurons, hidden_layers):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_1 = nn.Linear(input_shape, hidden_neurons)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_neurons, hidden_neurons) for _ in range(hidden_layers)]
        )

        self.classification = nn.Linear(hidden_neurons, num_classes)
        self.output = nn.Softmax(dim=-1)

    def forward(self, x):
        h = self.flatten(x)
        h = self.layer_1(h)
        for layer in self.hidden_layers:
            h = layer(h)
        h = self.classification(h)
        h = self.output(h)
        return h


class MLPClassifier(MLPBaseClassifier):
    def __init__(
        self,
        learning_rate=1e-3,
        class_weight="balanced",
        max_iter=100,
        hidden_neurons=8,
        hidden_layers=4,
        random_state=None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            class_weight=class_weight,
            max_iter=max_iter,
        )
        self.random_state = random_state
        self.hidden_neurons = hidden_neurons
        self.hidden_layers = hidden_layers

    def _initialize(self, X, y):
        torch.manual_seed(self.random_state)
        return super()._initialize(X, y)

    def _setup_model(self, num_classes, input_shape):

        self.model = MLPModel(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_neurons=self.hidden_neurons,
            hidden_layers=self.hidden_layers,
        )
        return self

    def _compute_loss(self, y, pred):
        if not hasattr(self, "loss"):
            self.loss = CrossEntropyLoss(weight=self._class_weight)

        loss = self.loss(pred, y)

        return loss
