import torch
import math

class AveragePrecisionMeter(object):

    def __init__(self):
        super(AveragePrecisionMeter, self).__init__()

        self.reset()

    def reset(self):

        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):

        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'

        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        offset = self.scores.size(0) if self.scores.dim() > 0 else 0

        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):

        if self.scores.numel() == 0:
            return 0

        ap = torch.zeros(self.scores.size(1))

        for k in range(self.scores.size(1)):

            scores = self.scores[:, k]
            targets = self.targets[:, k]

            ap[k] = AveragePrecisionMeter.average_precision(scores, targets)
        return ap

    @staticmethod
    def average_precision(output, target):

        sorted, indices = torch.sort(output, dim=0, descending=True)

        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if label == 0:
                total_count += 1
            if label == 1:
                pos_count += 1
                total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= (pos_count + 1e-10)
        return precision_at_i
