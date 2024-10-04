from .score import get_cell_scorer, grid_accuracy_scores, cell_accuracy_scores
from torchmetrics import AUROC, Accuracy, ConfusionMatrix, F1Score

accuracy_scores = cell_accuracy_scores
auroc_scores = get_cell_scorer(AUROC, 'auroc')
#confmat = get_cell_scorer(ConfusionMatrix, 'confmat')

