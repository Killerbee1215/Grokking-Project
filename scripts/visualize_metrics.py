import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_metrics(metrics_path):
    metrics = pd.read_csv(metrics_path)
    return metrics

def fit_curve(data):
    valid_idx = ~np.isnan(data)  
    x = np.arange(len(data))     
    y = data[valid_idx]          
    f = interp1d(x[valid_idx], y, kind='linear', fill_value="extrapolate")    
    
    return f(x)

# fill your own metrics path here
metrics_path = r"D:\grok-main\default\version_11\metrics.csv"
metrics = load_metrics(metrics_path)
train_acc = metrics['train_accuracy']
val_acc = metrics['val_accuracy']
step = metrics['step']

full_train_acc = np.zeros(max(step)+1)
full_val_acc = np.zeros(max(step)+1)
non_step = np.setdiff1d(np.arange(max(step)+1), step)
full_train_acc[step] = train_acc
full_val_acc[step] = val_acc
full_train_acc[non_step] = np.nan
full_val_acc[non_step] = np.nan

train_accuracy_fitted = np.apply_along_axis(fit_curve, axis=0, arr=full_train_acc)
test_accuracy_fitted = np.apply_along_axis(fit_curve, axis=0, arr=full_val_acc)

plt.plot(train_accuracy_fitted, label='train_accuracy')
plt.plot(test_accuracy_fitted, label='val_accuracy')
plt.xscale('log')
plt.xlabel('Optimization Steps')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('accuracy_curve.png')
