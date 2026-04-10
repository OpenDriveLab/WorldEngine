from mmcv.runner.hooks.hook import HOOKS, Hook

@HOOKS.register_module()
class GradChecker(Hook):
    def after_train_iter(self, runner):
        for key, val in runner.model.named_parameters():
            if val.grad == None and val.requires_grad:
                print(
                    "WARNNING: {key}'s parameters are not be used!!!!".format(key=key)
                )


@HOOKS.register_module()
class SamplerSkipIterationHook(Hook):
    """Data-loading sampler for distributed training.

    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def __init__(self, out_dir=None):
        """Init routine."""
        self.out_dir = out_dir

    def before_train_epoch(self, runner):
        if hasattr(runner.data_loader.sampler, 'skip_iter_at_epoch_x'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.data_loader.sampler.skip_iter_at_epoch_x(runner._inner_iter)
        elif hasattr(runner.data_loader.batch_sampler.sampler, 'skip_iter_at_epoch_x'):
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.data_loader.batch_sampler.sampler.skip_iter_at_epoch_x(runner._inner_iter)
