import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report, roc_curve, confusion_matrix,
    precision_score, recall_score, log_loss, f1_score, accuracy_score
)
from sklearn.tree import plot_tree
from sklearn.base import BaseEstimator
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

class ModelEvaluator:
    def __init__(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, th: float = 0.5):
        self.model = model
        self.X = X
        self.y = y
        self.y_pred = (self.model.predict_proba(self.X)[:, 1] > th).astype(int)
        self.y_probas = self.model.predict_proba(self.X)

    def model_report(self) -> pd.DataFrame:
        roc_auc = roc_auc_score(self.y, self.y_pred)
        print("ROC AUC:", roc_auc)
        class_report = classification_report(self.y, self.y_pred, output_dict=True)
        return pd.DataFrame(class_report)

    def evaluate(self) -> pd.DataFrame:
        auc = roc_auc_score(self.y, self.y_pred) * 100
        precision = precision_score(self.y, self.y_pred) * 100
        recall = recall_score(self.y, self.y_pred) * 100
        logloss = log_loss(self.y, self.y_pred) * 100
        f1 = f1_score(self.y, self.y_pred) * 100
        acc = accuracy_score(self.y, self.y_pred) * 100

        df = pd.DataFrame({
            'AUC': [auc],
            'PRECISION': [precision],
            'RECALL': [recall],
            'LOGLOSS': [logloss],
            'F1': [f1],
            'ACCURACY': [acc]
        })
        df = df.round(2)
        return df

    def plot_confusion_matrix(self, normalize: str = 'pred', return_matrix: bool = False) -> None:
        cm = confusion_matrix(self.y, self.y_pred, normalize=normalize)
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()

        if return_matrix:
            return cm
        else:
            return None

    def plot_roc(self) -> None:
        fpr, tpr, _ = roc_curve(self.y, self.y_pred)
        roc_auc = roc_auc_score(self.y, self.y_pred)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Model')
        roc_display.plot()
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

    def plot_lift(self):
        # Ordena as previsões e os rótulos reais
        data = pd.DataFrame({'y_true': self.y, 'y_probas': self.y_probas[:, 1]})
        data = data.sort_values(by='y_probas', ascending=False)
        data['cum_responders'] = data['y_true'].cumsum()

        # Calcula o lift para cada decil
        total_responders = data['y_true'].sum()
        data['lift'] = data['cum_responders'] / ((np.arange(len(data)) + 1) * total_responders / len(data))

        # Cria a curva de lift
        x_values = np.arange(1, len(data) + 1) / len(data)
        plt.plot(x_values, data['lift'], label='Lift Curve')
        plt.plot([0, 1], [1, 1], 'k--', label='Baseline')
        plt.xlabel('Percentage of Sample')
        plt.ylabel('Lift')
        plt.title('Lift Curve')
        plt.legend(loc='lower right')
        plt.show()