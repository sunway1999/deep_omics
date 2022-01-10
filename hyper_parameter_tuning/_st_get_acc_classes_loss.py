# works for keras NN

import numpy as np

def get_acc_classes(model, data_to_test, label_to_test, n_fold):
    y2_test_list = [y[0] for y in label_to_test]
    w_test_list = [1 if y == 0 else n_fold for y in y2_test_list]
    results = model.evaluate(data_to_test, label_to_test, sample_weight = np.array(w_test_list))
    yhat = model.predict(data_to_test)
    yhat_label = (model.predict(data_to_test) > 0.5).astype("int32")
    y2_test_list = label_to_test.tolist()
    po_ind = [i for i in range(len(y2_test_list)) if y2_test_list[i][0] == 1]
    first_dim = label_to_test.shape[0]
    yhat_label = yhat_label.reshape(first_dim, 1)
    acc_on_po = yhat_label[po_ind].mean()
    ne_ind = [i for i in range(len(y2_test_list)) if y2_test_list[i][0] == 0]
    acc_on_ne = 1 - yhat_label[ne_ind].mean()
    return results[0], results[1], results[2], results[3], acc_on_po, acc_on_ne
