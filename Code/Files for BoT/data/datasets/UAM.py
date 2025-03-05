# encoding: utf-8
"""
@author:  weijian
@contact: dengwj16@gmail.com
"""

import glob
import re
import xml.dom.minidom as XD
import os.path as osp

from .bases import BaseImageDataset


class UAM(BaseImageDataset):
    """
    UAM

    Dataset statistics:

    """

    dataset_dir = './ReID/crosswalks/crosswalk13new/'
    dataset_dir_test = './ReID/crosswalks/ccrosswalk13new/'
    def __init__(self, root='./data',
                 verbose=True, **kwargs):
        super(UAM, self).__init__()
        self.dataset_dir = root
        #DESCOMENTARself.dataset_dir = osp.join(root, self.dataset_dir)
        #self.train_track_path = osp.join(self.dataset_dir, 'train_track.txt')
        #self.test_track_path = osp.join(self.dataset_dir_test, 'test_track.txt')

        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir, 'image_train/')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_train/')
        #self.query_dir = osp.join(self.dataset_dir_test, 'image_query/')
        #self.gallery_dir = osp.join(self.dataset_dir_test, 'image_test/')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)#self._process_dir(self.gallery_dir, relabel=False,query=False)


        if verbose:
            print("=> AIC21 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        xml_dir = osp.join(self.dataset_dir, 'train_label.xml')

        import xml.etree.ElementTree as ET
        xmlp = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(xml_dir,parser=xmlp)
        root = tree.getroot()

        pid_container = set()

        for element in root.iter('Item'):
            pid = int(element.get('objectID'))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        #_, _, frame2trackID = self._process_track(path=self.train_track_path) #### Revised

        for element in root.iter('Item'):
            pid, camid = map(int, [element.get('objectID'), element.get('cameraID')[1:]])
            image_name = str(element.get('imageName'))
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            #trackid = frame2trackID[image_name]
            #dataset.append((osp.join(dir_path, image_name), pid, camid,trackid))
            dataset.append((osp.join(dir_path, image_name), pid, camid))

        """
        #info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')

        for element in range(len(info)):
            pid = int(info[element].getAttribute('vehicleID'))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}       

        dataset = []
        for element in range(len(info)):
            pid, camid = map(int, [info[element].getAttribute('vehicleID'), info[element].getAttribute('cameraID')[1:]])
            image_name = str(info[element].getAttribute('imageName'))
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((osp.join(dir_path, image_name), pid, camid))
        """

        return dataset
    def _process_dir_test(self, dir_path, relabel=False, query=True):
        #_, _, frame2trackID = self._process_track(path=self.test_track_path)
        if query:
            xml_dir = osp.join(self.dataset_dir_test, 'query_label.xml')
        else:
            xml_dir = osp.join(self.dataset_dir_test, 'test_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
        dataset = []
        for element in range(len(info)):
            camid = int(info[element].getAttribute('cameraID')[1:])
            image_name = str(info[element].getAttribute('imageName'))
            #Comentar en all
            pid = int(info[element].getAttribute('objectID'))
            # if query:
            #     trackid = -1
            # else:
            #     trackid = frame2trackID[image_name]
            #dataset.append((osp.join(dir_path, image_name), -1, camid,trackid))


            #dataset.append((osp.join(dir_path, image_name), pid, camid, trackid))
            dataset.append((osp.join(dir_path, image_name), pid, camid))
        return dataset

    def _process_track(self,path): #### Revised

        file = open(path)
        tracklet = dict()
        frame2trackID = dict()
        nums = []
        for track_id, line in enumerate(file.readlines()):
            curLine = line.strip().split(" ")
            nums.append(len(curLine))
            #  curLine = list(map(eval, curLine))
            tracklet[track_id] =  curLine
            for frame in curLine:
                frame2trackID[frame] = track_id
        return tracklet, nums, frame2trackID

    def _process_dir_testVeri(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d\d\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
