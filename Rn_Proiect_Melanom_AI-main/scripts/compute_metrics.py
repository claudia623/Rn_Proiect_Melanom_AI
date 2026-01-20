import csv
from collections import defaultdict

gt_path = 'data/test/ISBI2016_ISIC_Part3_Test_GroundTruth.csv'
pred_path = 'logs/predictions.csv'
out_path = 'results/final_metrics_per_class.json'

# load ground truth
gt = {}
with open(gt_path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row: continue
        name = row[0].strip()
        label = row[1].strip()
        try:
            gt[name if name.endswith('.jpg') else name + '.jpg'] = int(float(label))
        except:
            gt[name if name.endswith('.jpg') else name + '.jpg'] = None

# load predictions (use last prediction per file)
preds = {}
with open(pred_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname = row['filename'].strip()
        cls = row['classification'].strip().upper()
        prob = float(row.get('probability_malignant', 0) or 0)
        preds[fname] = (1 if cls in ('MALIGNANT','MAL') else 0, prob)

# match and compute confusion
TP = FP = TN = FN = 0
matched = 0
unmatched_preds = []
for fname, (p,labelprob) in preds.items():
    if fname not in gt:
        unmatched_preds.append(fname)
        continue
    y = gt[fname]
    matched += 1
    if y==1 and p==1:
        TP += 1
    elif y==1 and p==0:
        FN += 1
    elif y==0 and p==1:
        FP += 1
    elif y==0 and p==0:
        TN += 1

precision_mal = TP / (TP+FP) if (TP+FP)>0 else 0.0
recall_mal = TP / (TP+FN) if (TP+FN)>0 else 0.0
precision_ben = TN / (TN+FN) if (TN+FN)>0 else 0.0
recall_ben = TN / (TN+FP) if (TN+FP)>0 else 0.0

import json
res = {
    'matched_predictions': matched,
    'unmatched_predictions_count': len(unmatched_preds),
    'unmatched_examples': unmatched_preds[:20],
    'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
    'precision_malignant': round(precision_mal,4),
    'recall_malignant': round(recall_mal,4),
    'precision_benign': round(precision_ben,4),
    'recall_benign': round(recall_ben,4)
}

with open(out_path, 'w') as f:
    json.dump(res, f, indent=2)

print('Wrote', out_path)
print(json.dumps(res, indent=2))
