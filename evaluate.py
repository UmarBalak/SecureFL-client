# evaluate_fixed.py
import os
import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Get logger for this module
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test_cat, class_names=None):
    """
    Evaluates the model with proper logging integration
    """
    logger.info("="*70)
    logger.info("🔧 Starting model evaluation on test dataset...")
    
    def compute_metrics(X, y_cat, dataset_name):
        logger.info(f"📊 Computing metrics for {dataset_name}...")
        
        # Model evaluation
        loss, acc = model.evaluate(X, y_cat, verbose=0)
        logger.info(f"✅ {dataset_name} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        
        # Predictions
        logger.info(f"🎯 Computing predictions for {dataset_name}...")
        y_pred = model.predict(X, verbose=0)
        y_true = np.argmax(y_cat, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        
        # Metrics computation
        logger.info(f"📈 Computing detailed metrics for {dataset_name}...")
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        # Log results
        logger.info(f"📋 {dataset_name} Per-Class Metrics:")
        if class_names:
            for i, name in enumerate(class_names):
                logger.info(f"   Class {name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1-Score={f1[i]:.4f}")
        
        logger.info(f"📊 {dataset_name} Aggregate Metrics:")
        logger.info(f"   Macro: Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1-Score={macro_f1:.4f}")
        logger.info(f"   Weighted: Precision={weighted_precision:.4f}, Recall={weighted_recall:.4f}, F1-Score={weighted_f1:.4f}")
        
        # Save confusion matrix
        try:
            os.makedirs('plots', exist_ok=True)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names or range(len(cm)), 
                       yticklabels=class_names or range(len(cm)))
            plt.title(f'{dataset_name} Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plot_path = f'plots/{dataset_name.lower().replace(" ", "_")}_confusion_matrix.png'
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"💾 Confusion matrix saved: {plot_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save confusion matrix: {e}")
        
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
            'classification_report': classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0) if class_names else classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
    
    test_results = compute_metrics(X_test, y_test_cat, "Test Set")
    
    logger.info("✅ Model evaluation completed successfully!")
    logger.info("="*70)
    
    return {'test': test_results}
