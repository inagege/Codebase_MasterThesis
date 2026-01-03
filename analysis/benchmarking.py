import pandas as pd
import os

if __name__ == '__main__':
    results = pd.read_csv(os.path.joint('out', 'out', 'all_predictions.csv'))
    predictions = results['prediction'].tolist()
    labels = results['label'].tolist()
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(labels)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Total samples: {total}, Correct predictions: {correct}, Accuracy: {accuracy:.4f}")