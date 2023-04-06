from crslab.evaluator.metrics.base import AverageMetric

class AccuracyMetric(AverageMetric):
    @staticmethod
    def compute(predict, label) -> "AccuracyMetric":
        predict_goal = 1 if predict > 0.5 else 0
        return AccuracyMetric(int(predict_goal == label))
