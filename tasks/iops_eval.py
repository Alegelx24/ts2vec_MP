# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import json
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, precision_score, recall_score


def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0

    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)

    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0

    return new_predict

def reconstruct_label(timestamp, label):
    timestamp = np.asarray(timestamp, np.int64)
    index = np.argsort(timestamp)

    timestamp_sorted = np.asarray(timestamp[index])
    interval = np.min(np.diff(timestamp_sorted))

    label = np.asarray(label, np.int64)
    label = np.asarray(label[index])

    idx = (timestamp_sorted - timestamp_sorted[0]) // interval

    new_label = np.zeros(shape=((timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1,), dtype=int)
    new_label[idx] = label

    return new_label

def label_evaluation(result_file, delay=7):
    data = {'result': False, 'data': "", 'message': ""}

    

    result_df = pd.read_csv(result_file)

    y_true_list = []
    y_pred_list = []

    y_true = reconstruct_label(result_df["Timestamps"], result_df["Labels"])
    y_pred = reconstruct_label(result_df["Timestamps"], result_df["Predictions"])

    y_pred = get_range_proba(y_pred, y_true, delay)
    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

    try:
        fscore = f1_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
        precision = precision_score(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    except:
        data['message'] = "The 'predict' column can only contain 0 or 1."
        return json.dumps(data, ensure_ascii=False)

    data['result'] = True
    data['data'] = fscore
    data['precision'] = precision
    data['message'] = 'Calculation successful.'

    return json.dumps(data, ensure_ascii=False)

if __name__ == '__main__':
    result_file = "/Users/aleg2/Downloads/updated_prediction_loss_negative.csv"  
    delay = 7  
    print(label_evaluation(result_file, delay))
