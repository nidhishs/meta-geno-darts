class AverageMeter:
    def __init__(self):
        """ Computes and stores the average and current value """
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def precision(output, target, top_k=(1,)):
    """
    Computes precision@k for given batch: Percentage of samples for which ground truth is in top_k predicted classes.
    @param output : [batch_size, num_classes] Softmax probabilities per class.
    @param target : [num_classes] Discrete value for ground truth classes.
    @param top_k : tuple of k values.
    """
    max_k = max(top_k)
    batch_size = target.size(0)

    _, top_k_predicted_classes = output.topk(max_k, dim=-1, largest=True, sorted=True)
    top_k_predicted_classes_t = top_k_predicted_classes.t()
    correct = top_k_predicted_classes_t == target.view(1, -1).expand_as(top_k_predicted_classes_t)

    res = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdims=True)
        res.append((correct_k / batch_size))

    return res
