import os
from config import cfg
import argparse
from data.build_DG_dataloader import build_reid_test_loader
from model import make_model
from processor.part_attention_vit_processor import do_inference as do_inf_pat
from processor.ori_vit_processor_with_amp import do_inference as do_inf
from utils.logger import setup_logger

import numpy as np
from re_ranking import re_ranking
import torch
from torch.backends import cudnn
from utils.re_rankingGPS import re_rankingGPS
import xml.etree.ElementTree as ET
import csv

def read_xml(xml_dir):
    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(xml_dir,parser=xmlp)
    root = tree.getroot()
    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(xml_dir, parser=xmlp)
    root = tree.getroot()
    dataset=[]
    for element in root.iter('Item'):
        pid = int(element.get('rename')[:-4])
        image_name = int(element.get('imageName')[:-4])
        video= (element.get('video'))
        dataset.append([pid,image_name,video])
    return dataset
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
    parser.add_argument(
        "--xml_dir_gallery", default="./config/PAT.yml", help="path to config file", type=str
    )
    parser.add_argument(
        "--xml_dir_query", default="./config/PAT.yml", help="path to config file", type=str
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
    np.save("./Part-Aware-Transformer/qf.npy", qf)
    np.save("./Part-Aware-Transformer/gf.npy", gf)

    #inference(cfg, model, val_loader, num_query)
    q_g_dist = np.dot(qf, np.transpose(gf))
    q_q_dist = np.dot(qf, np.transpose(qf))
    g_g_dist = np.dot(gf, np.transpose(gf))
    csv_0905='GPSannotations/0905_May_sequence/GX010042-frame_gps_interp.csv'
    csv_2803='GPSannotations/2803_March_sequence/GX010040-frame_gps_interp.csv'
    csv_1807 = 'GPSannotations/01807_July_sequence/GX010045-frame_gps_interp.csv'
    csv_1807inv = 'GPSannotations/1807_July_inverse_sequence/GX010046-frame_gps_interp.csv'

    xml_dir_gallery=cfg.DATASETS.ROOT_DIR+ args.xml_dir_gallery
    gallery_rename=read_xml(xml_dir_gallery)
    xml_dir_query = cfg.DATASETS.ROOT_DIR +  args.xml_dir_query
    query_rename=read_xml(xml_dir_query)


    with open(csv_0905, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Omitir la primera fila (encabezado)
        gps_coordinates0905 = {row[0]: (float(row[1]), float(row[2])) for row in reader}
    with open(csv_2803, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Omitir la primera fila (encabezado)
        gps_coordinates2803 = {row[0]: (float(row[1]), float(row[2])) for row in reader}
    with open(csv_1807, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Omitir la primera fila (encabezado)
        gps_coordinates1807= {row[0]: (float(row[1]), float(row[2])) for row in reader}
    with open(csv_1807inv, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Omitir la primera fila (encabezado)
        gps_coordinates1807inv= {row[0]: (float(row[1]), float(row[2])) for row in reader}
    #gf_gps = [gps_coordinates_gallery[str(gallery_rename[i][1])] for i in range(len(gallery_rename))]
    gf_gps = []
    for i in range(len(gallery_rename)):
        if str(gallery_rename[i][2]) == '0905':
            gf_gps.append(gps_coordinates0905[str(gallery_rename[i][1])])
        elif str(gallery_rename[i][2]) == '2803':
            gf_gps.append(gps_coordinates2803[str(gallery_rename[i][1])])
        elif str(gallery_rename[i][2]) == '1807':
            gf_gps.append(gps_coordinates1807[str(gallery_rename[i][1])])
        elif str(gallery_rename[i][2]) == '1807inv':
            gf_gps.append(gps_coordinates1807inv[str(gallery_rename[i][1])])
    qf_gps = []
    for i in range(len(query_rename)):
        if str(query_rename[i][2]) == '0905':
            qf_gps.append(gps_coordinates0905[str(query_rename[i][1])])
        elif str(query_rename[i][2]) == '2803':
            qf_gps.append(gps_coordinates2803[str(query_rename[i][1])])
        elif str(query_rename[i][2]) == '1807':
            qf_gps.append(gps_coordinates1807[str(query_rename[i][1])])
        elif str(query_rename[i][2]) == '1807inv':
            qf_gps.append(gps_coordinates1807inv[str(query_rename[i][1])])
    re_rank_dist = re_rankingGPS(qf, gf, qf_gps, gf_gps,query_rename,gallery_rename)
    #re_rank_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

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
