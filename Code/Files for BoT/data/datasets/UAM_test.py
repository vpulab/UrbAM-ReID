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


class UAM_test(BaseImageDataset):
    """
    VR

    Dataset statistics:

    """


    def __init__(self, root='./data',
                 verbose=True, **kwargs):
        super(UAM_test, self).__init__()
        #cambiado para .bash,
        self.dataset_dir = root
        self.dataset_dir_test = root
        self.train_dir = osp.join(self.dataset_dir, 'image_train/')
        self.query_dir = osp.join(self.dataset_dir_test, 'image_query/')
        self.gallery_dir = osp.join(self.dataset_dir_test, 'image_test/')

        self._check_before_run()

        query = self._process_dir_test(self.query_dir, relabel=False,query=True)
        gallery = self._process_dir_test(self.gallery_dir, relabel=False)

        train = self._process_dir(self.train_dir, relabel=True)

        if verbose:
            print("=> VR loaded")
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
        tree = ET.parse(xml_dir,
            parser=xmlp)
        root = tree.getroot()

        pid_container = set()

        for element in root.iter('Item'):
            pid = int(element.get('objectID'))
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for element in root.iter('Item'):
            pid, camid = map(int, [element.get('objectID'), element.get('cameraID')[1:]])
            image_name = str(element.get('imageName'))
            if pid == -1: continue  # junk images are just ignored
            if relabel: pid = pid2label[pid]
            dataset.append((osp.join(dir_path, image_name), pid, camid))

        """
        #info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')

        for element in range(len(info)):
            pid = int(info[element].getAttribute('objectID'))
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

    def _process_dir_test(self, dir_path, relabel=False,query=False):
        if query:
            xml_dir = osp.join(self.dataset_dir_test, 'query_label.xml')
        else:
            xml_dir = osp.join(self.dataset_dir_test, 'test_label.xml')
        info = XD.parse(xml_dir).documentElement.getElementsByTagName('Item')
        dataset = []
        for element in range(len(info)):
            camid = int(info[element].getAttribute('cameraID')[1:])
            image_name = str(info[element].getAttribute('imageName'))
            dataset.append((osp.join(dir_path, image_name), -1, camid))

        return dataset

        return dataset
    def _process_dir_demo(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        img_paths.sort()
        pid_container = set()
        for img_path in img_paths:
            pid = 1
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = 1, 2
            if pid == -1: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
