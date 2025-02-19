import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from processor.part_attention_vit_processor import do_inference as do_inf_pat
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from utils.logger import setup_logger

import numpy as np
from utils.re_ranking import re_ranking
import torch
from torch.backends import cudnn
def extract_feature(model, dataloaders, num_query):
    features = []
    count = 0
    img_path = []

    for data in dataloaders:
        img, a, b,_,_ = data.values()
        #obtain values form dict data
        n, c, h, w = img.size()
        count += n
        ff = torch.FloatTensor(n, 768).zero_().cuda()  # 2048 is pool5 of resnet
        for i in range(2):
            input_img = img.cuda()
            outputs = model(input_img)
            f = outputs.float()
            ff = ff + f
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features.append(ff)
    features = torch.cat(features, 0)

    # query
    qf = features[:num_query]
    # gallery
    gf = features[num_query:]
    return qf, gf
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Training")
    parser.add_argument(
        "--config_file", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--track", default="./config/PAT.yml", help="path to config file", type=str
    )
    args = parser.parse_args()



    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("PAT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    model = make_model(cfg, cfg.MODEL.NAME, 0,0,0)
    model.load_param(cfg.TEST.WEIGHT)

    for testname in cfg.DATASETS.TEST:
        val_loader, num_query = build_reid_test_loader(cfg, testname)
        if cfg.MODEL.NAME == 'part_attention_vit':
            do_inf_pat(cfg, model, val_loader, num_query)
        else:
            do_inf(cfg, model, val_loader, num_query)
    with torch.no_grad():
        qf, gf = extract_feature(model, val_loader, num_query)

    # save feature
    qf=qf.cpu().numpy()
    gf=gf.cpu().numpy()
    np.save("./qf.npy", qf)
    np.save("./gf.npy", gf)

    #inference(cfg, model, val_loader, num_query)
    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))

    re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    indices = np.argsort(re_rank_dist, axis=1)[:, :100]
    #indices = np.argsort(re_rank_dist, axis=1)[:, :]

    m, n = indices.shape
    print('m: {}  n: {}'.format(m, n))
    with open(args.track, 'wb') as f_w:
        for i in range(m):
            write_line = indices[i] + 1
            write_line = ' '.join(map(str, write_line.tolist())) + '\n'
            f_w.write(write_line.encode())
    print(indices[0])
    print(indices.shape)
