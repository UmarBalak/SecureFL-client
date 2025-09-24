import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test_cat, class_names=None):
    """
    Evaluates the model on the training, validation, and test datasets for multiclass classification.
    
    Args:
        model: Trained Keras model.
        X_test, y_test_cat: Test data and one-hot encoded labels.
        class_names: List of class names for readable output (optional).
    Returns:
        dict: Evaluation results including loss, accuracy, per-class metrics, and confusion matrix.
    """
    print("\n" + "="*70)
    print("Evaluating model performance on all datasets...")
    
    def compute_metrics(X, y_cat, dataset_name):
        loss, acc = model.evaluate(X, y_cat, verbose=0)
        print(f"✅ {dataset_name} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        
        y_pred = model.predict(X, verbose=0)
        y_true = np.argmax(y_cat, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0) if class_names else classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        print(f"\n{dataset_name} Per-Class Metrics:")
        print("-" * 50)
        if class_names:
            for i, name in enumerate(class_names):
                print(f"Class {name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")
        else:
            for i in range(len(precision)):
                print(f"Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")
        
        print(f"\n{dataset_name} Aggregate Metrics:")
        print(f"Macro Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1-Score: {macro_f1:.4f}")
        print(f"Weighted Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1-Score: {weighted_f1:.4f}")
        
        # Ensure plots directory exists
        os.makedirs('plots', exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names or range(len(cm)), yticklabels=class_names or range(len(cm)))
        plt.title(f'{dataset_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'plots/{dataset_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.close()
        
        return {
            'loss': loss,
            'accuracy': acc,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1_score': f1.tolist(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
    
    test_results = compute_metrics(X_test, y_test_cat, "Test Set")
    results = {
        'test': test_results
    }
    
    return results
