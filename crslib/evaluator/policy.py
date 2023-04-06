
import time
from loguru import logger

from torch.utils.tensorboard import SummaryWriter

from crslab.evaluator.utils import nice_report
from crslab.evaluator.base import BaseEvaluator
from crslab.evaluator.metrics import Metrics, aggregate_unnamed_reports
from crslab.evaluator.metrics.base import SumMetric
from .metrics import AccuracyMetric

class PolicyEvaluator(BaseEvaluator):
    def __init__(self, tensorboard=False):
        super().__init__()
        self.acc_metrics = Metrics()
        self.optim_metrics = Metrics()
        self.tensorboard = tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir='runs/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            self.reports_name = ['Accuracy Metrics', 'Optimization Metrics']

    def acc_evaluate(self, predict, label, batch):
        self.acc_metrics.add(f"Accuracy", AccuracyMetric.compute(predict, label))
        if predict > 0.5:
            self.acc_metrics.add(f"Precision", AccuracyMetric.compute(predict, label))
        if label == 1:
            self.acc_metrics.add(f"Recall", AccuracyMetric.compute(predict, label))
        if label == 1 or predict > 0.3:
            self.acc_metrics.add(f"Hard Sample Accuracy", AccuracyMetric.compute(predict, label))
            self.acc_metrics.add(f"Hard Sample Number", SumMetric(1))

    def report(self, epoch=-1, mode='test'):
        reports = [self.acc_metrics.report(), self.optim_metrics.report()]
        if self.tensorboard and mode != 'test':
            for idx, task_report in enumerate(reports):
                for each_metric, value in task_report.items():
                    self.writer.add_scalars(f'{self.reports_name[idx]}/{each_metric}', {mode: value.value()}, epoch)
        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))

    def reset_metrics(self):
        self.acc_metrics.clear()
        self.optim_metrics.clear()