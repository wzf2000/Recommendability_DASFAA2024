
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
        self.F1 = None
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
        if 'Precision' in reports[0] and 'Recall' in reports[0]:
            Precision = reports[0]['Precision']
            Recall = reports[0]['Recall']
            if Precision.value() + Recall.value() > 0:
                self.F1 = 2 * Precision.value() * Recall.value() / (Precision.value() + Recall.value())
            else:
                self.F1 = 0
            logger.info(f'[Evaluation report: F1 = {self.F1}]')
        else:
            self.F1 = 0
        if self.tensorboard and mode != 'test':
            for idx, task_report in enumerate(reports):
                for each_metric, value in task_report.items():
                    self.writer.add_scalars(f'{self.reports_name[idx]}/{each_metric}', {mode: value.value()}, epoch)
        logger.info('\n' + nice_report(aggregate_unnamed_reports(reports)))
        if mode == 'test':
            acc = reports[0]['Accuracy'].value() if 'Accuracy' in reports[0] else 0
            precision = reports[0]['Precision'].value() if 'Precision' in reports[0] else 0
            recall = reports[0]['Recall'].value() if 'Recall' in reports[0] else 0
            F1 = self.F1 if self.F1 is not None else 0
            logger.info(f"\n{acc}\t{precision}\t{recall}\t{F1}")
            return [acc, precision, recall, F1]
        else:
            return []

    def reset_metrics(self):
        self.acc_metrics.clear()
        self.optim_metrics.clear()