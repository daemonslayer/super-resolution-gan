#!/usr/bin/env python

import sys
import os.path
import argparse
from argparse import RawTextHelpFormatter
import yaml
import time
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from mapper import mapper
from dataloader import SRDataset 
from pix2pix_trainer import Pix2PixTrainer

summary_writer = SummaryWriter('../runs/exp2_remote_run2')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    assert os.path.exists('./config.yaml'), '[!] Config file not found, exiting..'
    with open('config.yaml', 'r') as file:
        stream = file.read()
        config_dict = yaml.safe_load(stream)
        config = mapper(**config_dict)
    
    # print('GPU Ids: {}'.format(config.train.gpu_id.split(',')))
    trainer = Pix2PixTrainer(config.train)
    trainer.setup(config.train)
    total_iters = 0
    log_name = os.path.join(config.train.checkpoints.loc, config.train.title, 'loss_log.txt')

    # train_data = SRDataset(args=config.train.data, train=True,
    #                             transform=transforms.Compose([transforms.ToTensor()]))

    # test_data  = SRDataset(args=config.train.data, train=False,
    #                             transform=transforms.Compose([transforms.ToTensor()]))

    # val_data   = SRDataset(args=config.train.data, train=False,
    #                             transform=transforms.Compose([transforms.ToTensor()]))

    train_data = SRDataset('./', 
                    split='train', 
                    crop_size=96, 
                    scaling_factor=4, 
                    lr_img_type='imagenet-norm',
                    hr_img_type='[-1, 1]')
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=config.train.data.batch_size, shuffle=config.train.data.shuffle,
        num_workers=config.train.data.workers, pin_memory=config.train.data.pin_memory)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=config.train.data.batch_size, shuffle=config.train.data.shuffle,
        num_workers=config.train.data.workers, pin_memory=config.train.data.pin_memory)

    for epoch in range(config.train.hp.epoch_count, config.train.hp.n_epochs + config.train.hp.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        trainer.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_loader):  # inner loop within one epoch
            if data is None:
                continue
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % config.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += config.train.data.batch_size
            epoch_iter += config.train.data.batch_size
            trainer.set_input(data)         # unpack data from dataset and apply preprocessing
            trainer.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % config.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % config.update_html_freq == 0
                trainer.compute_visuals()
                visuals = trainer.get_current_visuals()
                image_viz = []
                for key, value in visuals.items():
                    image_viz.append(value[0])
                summary_writer.add_image('Fake_B/Real_A/Real_B', make_grid(image_viz), global_step=epoch)

            if total_iters % config.print_freq == 0:    # print training losses and save logging information to the disk
                losses = trainer.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config.train.data.batch_size
                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, epoch_iter, t_comp, t_data)
                for k, v in losses.items():
                    message += '%s: %.3f ' % (k, v)
                print(message)  # print the message
                with open(log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
                # summary_writer.add_scalar('Loss', losses, epoch)
                for key, value in losses.items():
                    summary_writer.add_scalar('{}_loss'.format(key), value, global_step=epoch)

            if total_iters % config.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if config.save_by_iter else 'latest'
                trainer.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % config.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            trainer.save_networks('latest')
            trainer.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, config.train.hp.n_epochs + config.train.hp.n_epochs_decay, time.time() - epoch_start_time))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
	parser.add_argument('--train', type=str2bool, default='1', \
				help='Turns ON training; default=ON')
	parser.add_argument('--test', type=str2bool, default='0', \
				help='Turns ON testing; default=OFF')
	args = parser.parse_args()
	main(args)