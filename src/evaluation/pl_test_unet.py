from src.utils import utils
from src.dataset_preproc.pl_datamodules import EndoDataModule
from src.pl_modules.unet import Unet, UnetGaussian

import os
import sys
import time
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import reset_seed, seed_everything

# Append project dir to sys path so other modules are available
current_dir = os.getcwd()  # ~/endoscopic-image-analysis/src/data
project_root = os.path.dirname(current_dir)  # ~/endoscopic-image-analysis
sys.path.append(project_root)

seed = 10
seed_everything(seed, workers=True)


def get_namespace(group, args):
    return argparse.Namespace(**dict((k, v) for k, v in vars(args).items() if k in group))


def main(args, config_dict):
    [gen_namespace, data_namespace, trainer_namespace] = \
        [get_namespace(group, args) for group in [config_dict['gen_arg_names'],
                                                  config_dict['data_arg_names'], config_dict['trainer_arg_names']]]

    data_config = utils.json_loader(data_namespace.data_config_path) if data_namespace.data_config_path else \
        {k: v for k, v in vars(data_namespace).items()}

    # Take care of device selection, on local/cluster
    gpus = [trainer_namespace.gpus] if gen_namespace.manual_select_gpu else trainer_namespace.gpus

    exp_name = gen_namespace.exp_name if gen_namespace.exp_name else "trial_run"

    tb_logger = TensorBoardLogger(save_dir=gen_namespace.log_dir,
                                  name=gen_namespace.model_name,
                                  version=exp_name,
                                  default_hp_metric=False)

    endo_data_module = EndoDataModule(**data_config)
    model = UnetGaussian.load_from_checkpoint(gen_namespace.model_checkpoint_path)
    trainer = Trainer.from_argparse_args(args,
                                         logger=tb_logger,
                                         gpus=gpus)
    trainer.test(model, endo_data_module)
    reset_seed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters")
    parser.add_argument("--model_checkpoint_path", type=str, required=True,
                        help="path to model checkpoint file")
    parser.add_argument("--model_name", type=str, default="suture_detection",
                        help="the name of the folder to save the test results in")
    parser.add_argument("--exp_name", type=str, default="",
                        help="the name of the experiment under the model")
    parser.add_argument("--log_dir", type=str, default=os.path.join(project_root, "results"),
                        help="log directory")
    parser.add_argument('--manual_select_gpu',
                        help="if set, then 'gpus' argument specified which device(s) to train on",
                        action="store_true")
    gen_arg_names = set(vars(parser.parse_known_args()[0]).keys())

    parser = EndoDataModule.add_data_specific_args(parser)
    data_arg_names = set(vars(parser.parse_known_args()[0]).keys()) - gen_arg_names

    parser = Trainer.add_argparse_args(parser)
    trainer_arg_names = set(vars(parser.parse_known_args()[0]).keys())-gen_arg_names-data_arg_names
    parsed_args = parser.parse_args()

    before_test_time = time.time()
    main(args=parsed_args, config_dict={'gen_arg_names': gen_arg_names,
                                        'data_arg_names': data_arg_names,
                                        'trainer_arg_names': trainer_arg_names})
    test_duration = time.time() - before_test_time

    hrs, minutes, sec = utils.get_hrs_min_sec(test_duration)
    print('Total test duration: {} hrs {} min {} sec'.format(hrs, minutes, sec))