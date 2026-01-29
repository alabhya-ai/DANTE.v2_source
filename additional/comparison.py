'''
Comparison of Different models with DANTE.v2.
We consider 'malicious' = +ve class and 'normal' = -ve class
'''

from math import ceil

def FAR(fp, N):
    '''False Alarm Rate (FAR) = fp / (fp + tn) = fp / N, where N = total actual negatives'''
    return (fp / N) * 100 # in percentage

# for DANTE.v2:
print("\nDANTE.v2:\n")

tp = 50
fp = 651
tn = 116251
fn = 410

DANTE_v2_FAR = FAR(fp = fp, N = fp + tn)

# we know other metrics from evaluate.ipynb, so no need to redo them
print(f"DANTE.v2's FAR: {DANTE_v2_FAR}%")
print(f"False Alarms on DANTE.v2's test set: ", fp)


# for DANTE (Original)

print("\nDANTE (Original):\n")
tp = 1
fp = 19
tn = 37297
fn = 65

testSetSize = tp + fp + tn + fn
DANTE_og_FAR = FAR(fp = fp, N = fp + tn)

# we know other metrics from evaluate_ogDANTE.ipynb, so no need to redo them
print(f"Original DANTE's FAR: {DANTE_og_FAR}%")

# Keeping same ratio in DANTE.v2's test set of 117362 entries:

tp = (tp / testSetSize) * 117362
fp = (fp / testSetSize) * 117362
tn = (tn / testSetSize) * 117362
fn = (fn / testSetSize) * 117362

print("False Alarms on DANTE.v2's test set if same performance is maintained:", ceil(fp))

# for Sharma et al.
print("\nSharma et al.:\n") # esko calculation chai preprocessingFiles.ipynb (experimenting4.2 folder) ma xa

# FPR = FAR = 0.0984
# FAR = fp / N
# N = 116902 in DANTE.v2's test set

# If the model maintains the same performance in DANTE.v2's test set:
# False Alarms = fp = FAR * N

# 340801 normal and 993 anomalous

rec = 0.9103 # recall
tp = rec * 993 # tpr = recall = tp / total actual positives
fp = 340801 * 0.0984 # fpr = fp / total actual negatives

prec = tp / (tp + fp)

f1 = 2 * prec * rec / (prec + rec)

print("Sharma et al.'s precision score:", prec)
print("Sharma et al.'s recall score:", rec)
print("Sharma et al.'s F-1:", f1)
print("Sharma et al.'s FAR: 9.84%")
print("False Alarms on DANTE.v2's test set if same performance is maintained:", ceil(0.0984 * 116902))

# for Nasir et al.
print("\nNasir et al.:\n")
# confusion matrix - [[180732, 4163], [14572, 529]]
tp = 529
fp = 4163
tn = 180732
fn = 14572

testSetSize = tp + fp + tn + fn
Nasir_et_al_FAR = FAR(fp = fp, N = fp + tn)

print(f"Nasir et al.'s FAR: {Nasir_et_al_FAR}%")

prec = tp / (tp + fp) # precision
rec = tp / (tp + fn) # recall
F1 = 2 * prec * rec / (prec + rec) # F1

print(f"Nasir et al.'s precision: {prec}%")
print(f"Nasir et al.'s recall: {rec}%")
print(f"Nasir et al.'s F-1: {F1}%")
print("False Alarms on DANTE.v2's test set if same performance is maintained:", ceil(Nasir_et_al_FAR * 116902 / 100))