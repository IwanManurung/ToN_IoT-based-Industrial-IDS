import numpy as np
from typing import List, Dict, Callable, Literal

class iids_evaluator:
    def __init__(self):
        self.__reset_evaluator()
        pass
    
    def __reset_evaluator(self):
        self.__cm = {
            "y_pred-y_true": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m1-y_true": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m2-y_true": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m3-y_true": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m4-y_true": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m5-y_true": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m6-y_true": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m1-m2": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m1-m3": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m2-m3": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m4-m5": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m4-m6": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            "m5-m6": {'tp': 0, 'tn':0, 'fp':0, 'fn':0},
            }
        
    @property
    def conf_matrix(self):
        return self.__cm
    
    @staticmethod
    def __calc_cm(yt, yp):
        tn = 0
        fp = 0
        fn = 0
        tp = 0
        c1 = yt == yp
        c2 = yp == 1

        if c1 & c2:
            tp = 1
        elif c1 & ~c2:
            tn = 1
        elif ~c1 & c2:
            fp = 1
        elif ~c1 & ~c2:
            fn = 1
        return (tn, fp, fn, tp)
    
    def __calc_metrics(self):
        metrics = {}
        for cm_name in self.__cm.keys():
            metrics.update({cm_name: {}})
            tp = self.__cm.get(cm_name).get('tp')
            tn = self.__cm.get(cm_name).get('tn')
            fp = self.__cm.get(cm_name).get('fp')
            fn = self.__cm.get(cm_name).get('fn')
            # acc
            divisor = tp + tn + fn + fp
            if divisor == 0:
                acc = np.nan
            else:
                acc = (tp + tn) / divisor
            del divisor

            # precision
            divisor = tp + fp
            if divisor == 0:
                precision = 0
            else:
                precision = tp / divisor
            del divisor

            # recall
            divisor = tp + fn
            if divisor == 0:
                recall = 0
            else:
                recall = tp / divisor
            del divisor

            # f1 score
            divisor = precision + recall
            if np.isnan(divisor) or (divisor == 0):
                f1 = 0
            else:
                f1 = 2 * precision * recall / divisor
            del divisor

            # mcc
            divisor = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            if divisor == 0:
                mcc = 0
            else:
                mcc = ((tp * tn) - (fp * fn)) / np.sqrt(float(divisor))
            del divisor

            # kappa
            divisor = ((tp + fp) * (fp + tn)) + ((tp + fn) * (fn + tn))
            if divisor == 0:
                kappa = 0
            else:
                kappa = 2 * ((tp * tn) - (fn * fp)) / divisor
            del divisor

            # update metrics with model vs groundtruth
            if 'y_true' in cm_name:
                metrics.get(cm_name).update({'Accuracy': 100*acc})
                metrics.get(cm_name).update({'F1 score': 100*f1})
                metrics.get(cm_name).update({'Precision': 100*precision})
                metrics.get(cm_name).update({'Recall': 100*recall})
                metrics.get(cm_name).update({'MCC': float(mcc)})
                metrics.get(cm_name).update({'Kappa': float(kappa)})
            else:
                # update metrics with models vs other models
                metrics.get(cm_name).update({'Kappa': float(kappa)})

        self.__metrics = metrics

    def run_metrics_calcs(self, runtime):
        self.__calc_metrics()
        self.__metrics.update({"runtime": runtime})
    
    @property
    def metrics(self):
        return self.__metrics
    
    def update(self, ouput_model, y_pred, y_true):
        cm_name = 'y_pred-y_true'
        tn, fp, fn, tp = self.__calc_cm(yt=y_true, yp=y_pred)
        self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
        self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
        self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
        self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
        del cm_name, tn, fp, fn, tp

        cm_name = 'm1-y_true'
        tn, fp, fn, tp = self.__calc_cm(yt=y_true, yp=ouput_model[0])
        self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
        self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
        self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
        self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
        del cm_name, tn, fp, fn, tp

        cm_name = 'm2-y_true'
        tn, fp, fn, tp = self.__calc_cm(yt=y_true, yp=ouput_model[1])
        self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
        self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
        self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
        self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
        del cm_name, tn, fp, fn, tp

        cm_name = 'm3-y_true'
        tn, fp, fn, tp = self.__calc_cm(yt=y_true, yp=ouput_model[2])
        self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
        self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
        self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
        self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
        del cm_name, tn, fp, fn, tp

        cm_name = 'm1-m2'
        tn, fp, fn, tp = self.__calc_cm(yt=ouput_model[0], yp=ouput_model[1])
        self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
        self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
        self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
        self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
        del cm_name, tn, fp, fn, tp

        cm_name = 'm1-m3'
        tn, fp, fn, tp = self.__calc_cm(yt=ouput_model[0], yp=ouput_model[2])
        self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
        self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
        self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
        self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
        del cm_name, tn, fp, fn, tp

        cm_name = 'm2-m3'
        tn, fp, fn, tp = self.__calc_cm(yt=ouput_model[1], yp=ouput_model[2])
        self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
        self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
        self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
        self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
        del cm_name, tn, fp, fn, tp

        if len(ouput_model) == 6:
            cm_name = 'm4-y_true'
            tn, fp, fn, tp = self.__calc_cm(yt=y_true, yp=ouput_model[3])
            self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
            self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
            self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
            self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
            del cm_name, tn, fp, fn, tp

            cm_name = 'm5-y_true'
            tn, fp, fn, tp = self.__calc_cm(yt=y_true, yp=ouput_model[4])
            self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
            self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
            self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
            self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
            del cm_name, tn, fp, fn, tp

            cm_name = 'm6-y_true'
            tn, fp, fn, tp = self.__calc_cm(yt=y_true, yp=ouput_model[5])
            self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
            self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
            self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
            self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
            del cm_name, tn, fp, fn, tp

            cm_name = 'm4-m5'
            tn, fp, fn, tp = self.__calc_cm(yt=ouput_model[3], yp=ouput_model[4])
            self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
            self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
            self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
            self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
            del cm_name, tn, fp, fn, tp

            cm_name = 'm4-m6'
            tn, fp, fn, tp = self.__calc_cm(yt=ouput_model[3], yp=ouput_model[5])
            self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
            self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
            self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
            self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
            del cm_name, tn, fp, fn, tp

            cm_name = 'm5-m6'
            tn, fp, fn, tp = self.__calc_cm(yt=ouput_model[4], yp=ouput_model[5])
            self.__cm.get(cm_name).update({'tp': tp + self.conf_matrix.get(cm_name).get('tp')})
            self.__cm.get(cm_name).update({'fn': fn + self.conf_matrix.get(cm_name).get('fn')})
            self.__cm.get(cm_name).update({'fp': fp + self.conf_matrix.get(cm_name).get('fp')})
            self.__cm.get(cm_name).update({'tn': tn + self.conf_matrix.get(cm_name).get('tn')})
            del cm_name, tn, fp, fn, tp