#!/usr/bin/env python
# coding=utf-8
import os
import cv2
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/gpu.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='predicted_file/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    file_set=save_folder+'file_set.txt'
    core_annotion = save_folder+'det_test_带电芯充电宝.txt'
    coreless_annotion=save_folder+'det_test_不带电芯充电宝.txt'
    if os.path.exists(core_annotion):
        os.remove(core_annotion)
    if os.path.exists(coreless_annotion):
        os.remove(coreless_annotion)
    if os.path.exists(file_set):
        os.remove(file_set)
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        img_name = os.path.basename(img_id).split('.')[0]
        with open(file_set,'a+') as fi:
            fi.write(img_name+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data

        im_det = img.copy()
        h, w, _ = im_det.shape




        need_save = False
        # scale each detection back up to the image
        scale = torch.Tensor([w, h, w, h])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.01:
                b=detections[0, i, j, 0].item()
                item = (detections[0, i, j, 1:]*scale).cpu().numpy()
                item = [int(n) for n in item]
                chinese = labelmap[i-1]
                #img_name=os.path.basename(img_id).split('.')[0]
                #print('img:'+img_name+' '+str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3]))
                # print(chinese+'gt\n\n')
                if chinese[0] == '带':
                    with open(core_annotion,'a+') as f:
                        f.write(img_name+' '+str(b)+' '+str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n');
                    chinese = 'Battery_Core'
                else:
                    with open(coreless_annotion,'a+') as f:
                        f.write(img_name+' '+str(b)+' '+str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+'\n');
                    chinese = 'Battery_Coreless'
                if b>=0.5:
                    cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 255, 255), 2)
                    cv2.putText(im_det, chinese, (item[0], item[1] - 5), 0, 0.6, (0, 255, 255), 2)



                need_save = True
                j += 1

        if need_save:
            dst_path = img_id.replace('Image', 'ImageTarget')
            dst_dir = os.path.dirname(dst_path)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            cv2.imwrite(dst_path, im_det)


def test(img_dir,anno_dir):
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model,map_location='cpu'))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(img_dir,anno_dir, target_transform=VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test('','')
