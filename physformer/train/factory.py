from .physformer_exp import PhysFormerExperiment
from .pretrain_exp import PretrainExperiment
from .baseline_exp import BaselineExperiment


def create_experiment(args):
    if args.model in ("PhysFormer", "PhysFormer-iGT"):
        return PhysFormerExperiment(args)
    return BaselineExperiment(args)


def create_pretrain_experiment(args):
    return PretrainExperiment(args)
