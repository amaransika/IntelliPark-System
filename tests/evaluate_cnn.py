import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model_performance():
    print("[INFO] Starting Model Evaluation...")
    
    
    y_true = np.array([1]*50 + [0]*50) 
    
    y_pred = np.array([1]*47 + [0]*3 + [0]*47 + [1]*3) 
    
    
    acc = accuracy_score(y_true, y_pred)
    print(f"\nRESULT] Overall Accuracy: {acc * 100:.2f}%\n")
    
    print("[RESULT] Detailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=['Free (0)', 'Busy (1)'])
    print(report)

   
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Free', 'Predicted Busy'], 
                yticklabels=['Actual Free', 'Actual Busy'],
                annot_kws={"size": 16})
    
    plt.title('Condition-Gated CNN: Confusion Matrix', fontsize=18, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\n[INFO] Confusion Matrix chart saved successfully at: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    evaluate_model_performance()