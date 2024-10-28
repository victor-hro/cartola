from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_normalized_class_weights(y):
    """
    Calcula os pesos normalizados para cada classe.
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y
    )

    return dict(zip(classes, weights))