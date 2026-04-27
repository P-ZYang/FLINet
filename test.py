import argparse

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from Config.dataset import *
from Config.metrics import *
import time
from collections import OrderedDict
import torch
from model.FLINet import Ours as FLINet

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser(description="IRSTD Testing")
parser.add_argument('--ROC_thr', type=int, default=10, help='num')
parser.add_argument("--model_names", default=['Ours'], type=list,
                    help="model_name: 'ACM', 'Ours01', 'DNANet', 'ISNet', 'ACMNet', 'Ours01', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=['IRSTD-1K/FLINet_IRSTD-1K_best.pth.tar'], type=list)
parser.add_argument("--dataset_dir", default=r'./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['IRSTD-1K'], type=list,
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--save_img", default=False, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default=r'./Result',
                    help="path of saved image")
parser.add_argument("--save_log", type=str, default=r'./Test/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()

def Test():
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    IoU = mIoU()
    nIoU = SamplewiseSigmoidMetric(nclass=1, score_thresh=0)
    pd_fa = PD_FA()
    roc = ROCMetric(nclass=1, bins=10)

    model = FLINet(mode='test')
    state_dict = torch.load(opt.pth_dir, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict['state_dict'].items():
        name = k[6:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
        new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    model.load_state_dict(new_state_dict)
    model.eval()
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for batch_idx, (img, gt_mask, size, img_dir) in enumerate(tbar):
            pred = model.forward(img)
            pred = pred[:, :, :size[0], :size[1]]
            gt_mask = gt_mask[:, :, :size[0], :size[1]]

            IoU.update((pred > 0.5), gt_mask)
            nIoU.update(pred, gt_mask)
            pd_fa.update((pred[0, 0, :, :] > opt.threshold).cpu(), gt_mask[0, 0, :, :], size)
            roc.update(pred, gt_mask)

            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0, 0, :, :]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(
                    opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')


        pixAcc, IOU = IoU.get()
        nIoU = nIoU.get()
        # # Pd Fa
        results2 = pd_fa.get()
        ture_positive_rate, false_positive_rate, recall, precision, FP, F1_score = roc.get()

        print('pixAcc: %.4f| IoU: %.4f | nIoU: %.4f | Pd: %.4f| Fa: %.4f |F1: %.4f'
              % (pixAcc * 100, IOU * 100, nIoU * 100, results2[0] * 100, results2[1] * 1e+6, F1_score * 100))


if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                Test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    # if dataset_name in pth_dir and model_name in pth_dir:
                    opt.test_dataset_name = dataset_name
                    opt.model_name = model_name
                    opt.train_dataset_name = pth_dir.split('/')[0]
                    print(pth_dir)
                    opt.f.write(pth_dir)
                    print(opt.test_dataset_name)
                    opt.f.write(opt.test_dataset_name + '\n')
                    opt.pth_dir = opt.save_log + pth_dir
                    Test()
                    print('\n')
                    opt.f.write('\n')
        opt.f.close()