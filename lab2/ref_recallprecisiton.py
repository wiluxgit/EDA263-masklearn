class MultiScore:
    def __init__(self, PRCurve=None, F1Score=None):
        self.PRCurve = PRCurve
        self.Recall = PRCurve[0]
        self.Precision = PRCurve[1]
        
def getCurves(name2fitted, dataSet=None):
    ret = {}
    for name, pipe in name2fitted.items():
        y_raw = dataSet.Y
        y_pred = pipe.predict(dataSet.X)
        y_binary = y_raw.replace({'>50K':1, '<=50K':0})
        y_prob = pipe.predict_proba(dataSet.X)[:,1]
        
        ret[name] = MultiScore(
            PRCurve=sk.metrics.precision_recall_curve(
                y_binary, y_prob
            ),
            F1Score=sk.metrics.f1_score(
                y_raw, y_pred, pos_label=">50K"
            )
        )
    return ret