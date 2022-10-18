from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class Test_runner:
    def test_metrics(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]
        assert accuracy_score(y_true, y_pred) == 0.5
        accuracy = lambda a, b: sum(i == j for i, j in zip(a, b)) / len(a)
        assert accuracy(y_pred, y_true) == 0.5
