
import tensorboardX
import argparse

from core.utils import prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
from core.trainer import *
from datasets import *

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.backends.cudnn as cudnn
import torch
import os
import sys
import shutil

import time

import torch.distributed as dist

startTime = time.time()

parser = argparse.ArgumentParser(description='PyTorch training')
parser.add_argument('--config', type=str, default='core/tiny_munit.yaml',
                        help='Path to the MUNIT config file.')
parser.add_argument('--output_path', type=str, default='/home/YOUR_PATH/data/kdd2023/models/results',
                        help="Path where images/checkpoints will be saved")
parser.add_argument("--resume", action="store_true",
                        help='Resumes training from last avaiable checkpoint')
parser.add_argument('--dataset', type=str, default='CCMNIST1',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--batch', type=int, default=1,
                        help="Batch size")
parser.add_argument('--env', type=int, default=0,
                        help="env not including")
parser.add_argument('--step', type=int, default=12,
                        help="training step 1 or 2 or cotraining:12")
parser.add_argument('--input_path', type=str, default='/data/YOUR_PATH/kdd2023/models/NYPD/pretrain_env0_step1/outputs/tiny_munit/checkpoints',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--input_path1', type=str, default='/data/YOUR_PATH/kdd2023/models/NYPD/pretrain_env0_step1/outputs/tiny_munit/checkpoints',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--input_path2', type=str, default='/data/YOUR_PATH/kdd2023/models/NYPD/pretrain_env0_step2/outputs/tiny_munit/checkpoints',
                        help="Path where images/checkpoints will be saved")
parser.add_argument('--device', type=str, default='7',
                        help="CUDA DEVICE")

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] =args.device


cudnn.benchmark = True

# Load experiment setting
config = get_config(args.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = args.output_path
config['gen']['dataset'] = args.dataset
config['dis']['dataset'] = args.dataset

# Setup model and data loader
device = torch.device('cuda')
os.environ['WORLD_SIZE'] = '4'
os.environ['RANK']= '0'
os.environ['MASTER_ADDR']= 'localhost'
os.environ['MASTER_PORT']= '12345'

if args.step == 1:
    trainer1 = torch.nn.DataParallel(MUNIT_Trainer1(config))
elif args.step == 2:
    trainer1 = torch.nn.DataParallel(MUNIT_Trainer1(config))
    trainer2 = torch.nn.DataParallel(MUNIT_Trainer2(config))
    trainer2.to(device)
elif args.step == 12:
    trainer1 = torch.nn.DataParallel(MUNIT_Trainer(config))
trainer1.to(device)


if args.dataset == "CCMNIST1":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_CCMNIST1_loaders(args.env, args.batch)
elif args.dataset == "FairFace":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_FairFace_loaders(args.env, args.batch)
elif args.dataset == "YFCC":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_YFCC_loaders(args.env, args.batch)
elif args.dataset == "NYPD":
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_NYPD_loaders(args.env, args.batch)




train_display_images_a = torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda()
test_display_images_a = torch.stack([test_loader_a.dataset[-1-i][0] for i in range(display_size)]).cuda()
test_display_images_b = torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(args.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(args.output_path + "/logs", model_name))
output_directory = os.path.join(args.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(args.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
if args.step == 1:
    iterations = trainer1.module.resume(checkpoint_directory, hyperparameters=config) if args.resume else 0
    trainer = trainer1
elif args.step == 12:
    iterations = trainer1.module.resume(checkpoint_directory, hyperparameters=config) if args.resume else 0
    trainer = trainer1
    # trainer1.module.resume1(args.input_path1, hyperparameters=config)
    # trainer1.module.resume2(args.input_path2, hyperparameters=config)
elif args.step == 2:
    _ = trainer1.module.resume(args.input_path, hyperparameters=config)
    iterations = trainer2.module.resume(checkpoint_directory, hyperparameters=config) if args.resume else 0
    trainer = trainer2
while True:
    for it, (a, b) in enumerate(zip(train_loader_a, train_loader_b)):
        images_a, y_a, z_a = a
        images_b, y_b, z_b = b
        trainer.module.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()
        y_a, y_b = y_a.cuda(), y_b.cuda()
        z_a, z_b = z_a.cuda(), z_b.cuda()

        # with Timer("Elapsed time in update: %f"):
            # Main training code
        if args.step == 1:
            trainer.module.dis_update(images_a, images_b, config)
            trainer.module.gen_update(images_a, images_b, config)
        elif args.step == 12:
            trainer.module.dis_update(images_a, images_b, config)
            trainer.module.gen_update(images_a, z_a, images_b, z_b, config)
        elif args.step == 2:
            ca_a1, s_a1 = trainer1.module.gen_a.encode(images_a)
            ca_b1, s_b1 = trainer1.module.gen_b.encode(images_b)
            ca_a2, s_a2 = trainer1.module.gen_a.encode(images_a)
            ca_b2, s_b2 = trainer1.module.gen_b.encode(images_b)
            trainer.module.dis_update(ca_a1, ca_b1, config)
            trainer.module.gen_update(ca_a2, z_a, ca_b2, z_b, config)
        torch.cuda.synchronize()
        write_loss(iterations, trainer, train_writer)


        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            train_writer.flush()

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0 and args.dataset != 'NYPD':
            with torch.no_grad():
                if args.step == 1 or args.step == 12:
                    test_image_outputs = trainer.module.sample(test_display_images_a, test_display_images_b)
                elif args.step == 2:
                    test_image_outputs = trainer1.module.sample2(test_display_images_a, test_display_images_b, trainer)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.module.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

train_writer.close()
endTime = time.time()

print("Running Time:", (endTime-startTime)/3600, "hours")