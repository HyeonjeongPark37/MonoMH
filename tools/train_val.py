import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse
import wandb

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester


parser = argparse.ArgumentParser(description='implementation of MonoMH')
parser.add_argument('--config', default='lib/kitti.yaml', dest='config', help='settings of detection in yaml format')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-t', '--test', dest='test', action='store_true', help='evaluate model on test set')
parser.add_argument('--work-date', default='test', type=str, help='output_path, date')
parser.add_argument('--work-dir', default='test', type=str, help='output_path, dir')
parser.add_argument('--save-path', default='outputs/', type=str, help='save path for output (defualt: outputs/)')

args = parser.parse_args()


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def main():
    # load cfg
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    output_path = os.path.join(args.save_path, args.work_date, args.work_dir)
    os.makedirs(output_path, exist_ok=True)

    mm = "bsz_" + str(cfg['dataset']['batch_size']) + "_lr_" + str(cfg['optimizer']['lr']) + "_" + str(cfg['optimizer']['type'])
    log_file = os.path.join(output_path, mm)
    logger = create_logger(log_file+'_train.log')

    if not args.evaluate and not args.test:
        wandb.init(project=args.work_date, dir=output_path)
        wandb.run.name = args.work_date + "_" + args.work_dir
        wandb.config.update(args)
        wandb.run.save()

    
    #  build dataloader
    train_loader, val_loader, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)
    if not args.evaluate and not args.test:
        wandb.watch(model)

    # evaluation mode
    if args.evaluate:
        tester = Tester(cfg['tester'], cfg['dataset'], model, val_loader, logger)
        tester.test()
        
        return

    if args.test:
        tester = Tester(cfg['tester'], cfg['dataset'], model, test_loader, logger)
        tester.test()
        return

    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)

    # build lr & bnm scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg,
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=val_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      output_path=output_path)
    trainer.train()


if __name__ == '__main__':
    main()
