import os.path
import time

from torch.utils.tensorboard import SummaryWriter


class DataLoger:
    def __init__(self, root_dir, exp_name):
        self.root_dir = root_dir
        self.exp_name = exp_name
        self.writer = SummaryWriter(log_dir=os.path.join(root_dir, exp_name))
        self.loss_path = os.path.join(os.path.join(root_dir, exp_name), 'loss_log.txt')
        with open(self.loss_path, 'a') as f:
            now = time.strftime("%c")
            f.write('================ Training Loss (%s) ================\n' % now)

    def log_loss(self, losses, epoch, steps,epoch_steps):
        with open(self.loss_path, 'a') as f:
            f.write('[epoch:%-4d, step:%-6d]' % (epoch, epoch_steps))
            for k, v in losses.items():
                f.write('%s: %-4.3f ' % (k, v))
            f.write('\n')
        for k, v in losses.items():
            self.writer.add_scalar(k, v, global_step=steps)
