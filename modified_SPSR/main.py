import os
import argparse
import torch
import random
import numpy as np
from shutil import copyfile
from src.config import Config
from src.edge_match import EdgeMatch
from src.create_data_list import create_data_list

def load_config(mode = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "--checkpoints", type = str, default = "./checkpoints", help = "model checkpoint path, default = ./checkpoints")
    parser.add_argument("--model", type = int, choices = [1, 2, 3], help = "1: edge model, 2: SR model, 3: joint SR model with edge enhancer")
    parser.add_argument("--train_img_path", type = str, default = "./train_images")
    parser.add_argument("--test_img_path", type = str, default = "./test_images")
    parser.add_argument("--eval_img_path", type = str, default = "./eval_images")

    if mode == 2:
        #parser.add_argument("--input", type = str, help = "path to a test image")
        parser.add_argument("--output", type = str, help = "path to a output folder")

    args = parser.parse_args()

    create_data_list(args.train_img_path, args.test_img_path, args.eval_img_path, "./list_folder")

    config_path = os.path.join(args.path, "config.yaml")

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if not os.path.exists(config_path):
        copyfile('./config.yaml', config_path)

    config = Config(config_path)

    #train mode
    if mode == 1:
        config.MODE = 1
    
    #test mode
    elif mode == 2:
        config.MODE = 2
    
    #eval mode
    elif mode == 3:
        config.MODE = 3

    return config


    

def main(mode = None):

    config = load_config(mode)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in config.GPU)

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmarck = True
    else:
        config.DEVICE = torch.device("cpu")

    #fix random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    model = EdgeMatch(config)
    model.load()

    # model training
    if config.MODE == 1:
        config.print()
        print("Start training...")
        model.train()

    # model test
    elif config.MODE == 2:
        print("Start testing...")
        model.test()

    # eval mode
    else:
        print("Start eval...")
        model.eval()


if __name__ == "__main__":
    main()
