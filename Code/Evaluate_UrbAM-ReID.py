import os
import numpy as np
import xml.dom.minidom as XD
import xml.etree.ElementTree as ET
import os.path as osp
import argparse
import sys
from io import StringIO

#function to read the xml file
def read_xml(xml_dir,dir_path):
    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(xml_dir,parser=xmlp)
    root = tree.getroot()
    xmlp = ET.XMLParser(encoding="utf-8")
    tree = ET.parse(xml_dir, parser=xmlp)
    root = tree.getroot()
    dataset=[]
    for element in root.iter('Item'):
        pid = int(element.get('objectID'))
        image_name = int(element.get('imageName')[:-4])
        dataset.append([pid,image_name])
    return dataset



# Load the ground truth of the query and gallery images (camID, objectID)
parser = argparse.ArgumentParser(description="ReID Eval")

parser.add_argument(
    "--track", default="./track_list.txt", help="path to config file", type=str
)

parser.add_argument(
    "--path", default="../ReID/crosswalks/crosswalk1/", help="path to config file", type=str
)
args = parser.parse_args()

xml_dir = os.path.join(args.path,'test_label.xml') 
dir_path=  args.path
dataset_galley=read_xml(xml_dir,dir_path)
gallery_gt=np.array(dataset_galley)
xml_dir = os.path.join(args.path,'query_label.xml')
dir_path=  os.path.join(args.path,'query_test')
dataset_query=read_xml(xml_dir,dir_path)
query_gt=np.array(dataset_query)

names_query =args.track

#read the track2 file
with open(names_query) as f:
    line = f.readlines()
    line = [x.strip() for x in line]
    track2 = []
    for i in range(len(line)):
        track2.append(line[i].split(' '))
    track2 = np.array(track2)
    track2 = track2.astype(np.int32)

#save the vehicle ID ground truth of the gallery
id_gallery = gallery_gt[:,0]
#save the vehicle ID ground truth of the gallery
id_query = query_gt[:,0]

AP=0.0
CMC=np.zeros( len(track2))

#Calculate AP and CMC for each query
for j in range(len(query_gt)):

    query_id = query_gt[j,0]
    true_positives = np.where(id_gallery == query_id)
    if len(true_positives[0]) == 0:
        a=1
    index = track2[j,:] - 1
    sortID = id_gallery[index]

    sortIDGallery = sortID
    #find the good index
    rows_good = np.where(sortIDGallery == query_id)

    #initialize
    ap=0
    cmc=np.zeros( len(track2))

    #find the number of good index
    ngood = (true_positives[0].shape)[0]

    if rows_good[0].size != 0:

        cmc[rows_good[0][0]:] = 1

        for i in range(len(rows_good[0])):
            #recall
            d_recall = 1.0 / ngood
            #precision
            precision = (i+1) * 1.0 / (rows_good[0][i]+1)
            if rows_good[0][i] != 0:
                old_precision =(i+1) * 1.0 / (rows_good[0][i]+1)
            else:
                old_precision = 1.0
            ap = ap + d_recall * (old_precision + precision) / 2

    CMC = CMC + cmc
    AP = AP + ap

#calculate de mean average precision
mAP = AP / len(track2)
#calculate the mean cumulative match characteristic
CMC = CMC / len(track2)
#print the results
print("mAP= %.14f" % mAP)
print("CMC-1 =%.14f" % CMC[0])
print("CMC-5 =%.14f" % CMC[4])
print("CMC-10 =%.14f" % CMC[9])
print("CMC-15 =%.14f" % CMC[14])
print("CMC-20 =%.14f" % CMC[19])
print("CMC-30 =%.14f" % CMC[29])
print("CMC-45 =%.14f" % CMC[44])
print("CMC-100 =%.14f" % CMC[ len(track2)-1])
