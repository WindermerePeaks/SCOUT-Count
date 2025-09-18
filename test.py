import os
import torch
import torchvision.transforms as standard_transforms
import misc.transforms as own_transforms
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from models.SCOUTCounter import SCOUTCount

torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

mean_std = ([MEAN1, MEAN2, MEAN3], [SIGMA1, SIGMA2, SIGMA3]) # input train set statistics

img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
restore = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        # standard_transforms.ToPILImage()
    ])
pil_to_tensor = standard_transforms.ToTensor()
LOG_PARA = 100.0  #密度图-实际人数，缩放因子
use_mask = True
CAM_THRS=0.2

dataRoots = [
    '/home/liuyanhuang/SCOUT_Count/data/CARPK/test',
    # '/home/liuyanhuang/SCOUT_Count/data/PUCPR/test',
    # '/home/liuyanhuang/SCOUT_Count/data/RSOC/ship/test',
    # '/home/liuyanhuang/SCOUT_Count/data/RSOC/large-vehicle/test',
]
datanames = ["CARPK"]
# datanames = ["PUCPR"]
# datanames = ["RSOC"]
# datanames = ["NWPU_MOC"]

model_path='/home/liuyanhuang/SCOUT_Count/pretrained_models/parameters.pth'

def main():
    for dataRoot, dataname in zip(dataRoots, datanames):
        file_list = [filename for root,dirs,filename in os.walk(dataRoot+'/img/')]                                           
        test(file_list[0], model_path, dataRoot, dataname)
    

def test(file_list, model_path, dataRoot, dataname):

    net = SCOUTCount("SCOUTCount")
    net.cuda()
    net.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False) #加载预训练模型权重
    net.eval()

    gts = []
    preds = []
    maes = AverageMeter()
    mses = AverageMeter()

    for filename in file_list:
        print(filename)
        imgname = os.path.join(dataRoot, 'img', filename)
        filename_no_ext = filename.split('.')[0]

        # denname = dataRoot + '/den/' + filename_no_ext + '.csv'
        # den = pd.read_csv(denname, sep=',',header=None).values
        # den = den.astype(np.float32, copy=False)
        denname = dataRoot + '/den/' + filename_no_ext + '.npy'
        den = np.load(denname)

        simname = dataRoot + '/sim_cliprs_npy/' + filename_no_ext + '.npy'
        sim = np.load(simname)

        img = Image.open(imgname)
        if img.mode == 'L': 
            img = img.convert('RGB')
        img = img_transform(img) 
        img = img.unsqueeze(0).cuda() 
        gt = np.sum(den)

        with torch.no_grad():
            pred_map_result = net.test_forward(img)
        if use_mask:
            sim = torch.from_numpy(sim).float()  
            mask = torch.where(sim > CAM_THRS, torch.ones_like(sim), sim).cuda()
            t_pred_map_result = pred_map_result * mask
            pred_map = t_pred_map_result.cpu().data.numpy()[0,0,:,:]
        else:
            pred_map = pred_map_result.cpu().data.numpy()[0,0,:,:]

        pred = np.sum(pred_map)/LOG_PARA
        pred_map = pred_map/LOG_PARA
        den = den/np.max(den+1e-20)

        fp="carpk"
        density_save_dir = "./"+fp +"/"+ dataname + "/density_ours/"
        density_gtsave_dir = "./"+fp +"/"+ dataname + "/gt/"
        density_imgsave_dir = "./"+fp +"/"+dataname + "/img/"

        if not os.path.exists(density_imgsave_dir):
            os.makedirs(density_imgsave_dir)
        if not os.path.exists(density_save_dir):
            os.makedirs(density_save_dir)
        if not os.path.exists(density_gtsave_dir):
            os.makedirs(density_gtsave_dir)

        # print(f"[{filename[0]}] pred_map min: {pred_map.min():.4f}, max: {pred_map.max():.4f}, sum: {pred_map.sum():.4f}")
        pred_frame = plt.gca()
        plt.imshow(pred_map, 'jet')
        pred_frame.axes.get_yaxis().set_visible(False)
        pred_frame.axes.get_xaxis().set_visible(False)
        pred_frame.spines['top'].set_visible(False) 
        pred_frame.spines['bottom'].set_visible(False) 
        pred_frame.spines['left'].set_visible(False) 
        pred_frame.spines['right'].set_visible(False) 
        plt.savefig(density_save_dir+filename_no_ext+'_pred_'+str(float(pred))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=300)

        plt.close()

        den_frame = plt.gca()
        plt.imshow(den, 'jet')
        den_frame.axes.get_yaxis().set_visible(False)
        den_frame.axes.get_xaxis().set_visible(False)
        den_frame.spines['top'].set_visible(False) 
        den_frame.spines['bottom'].set_visible(False) 
        den_frame.spines['left'].set_visible(False) 
        den_frame.spines['right'].set_visible(False) 
        plt.savefig(density_gtsave_dir+filename_no_ext+'_gt_'+str(int(gt))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=300)
        
        plt.close()

        img_frame = plt.gca()
        img = restore(img.squeeze(0)) 
        plt.imshow(img.permute(1,2,0).cpu().numpy())
        img_frame.axes.get_yaxis().set_visible(False)
        img_frame.axes.get_xaxis().set_visible(False)
        img_frame.spines['top'].set_visible(False) 
        img_frame.spines['bottom'].set_visible(False) 
        img_frame.spines['left'].set_visible(False) 
        img_frame.spines['right'].set_visible(False) 
        plt.savefig(density_imgsave_dir+filename_no_ext+'_img_'+str(int(gt))+'.png',\
            bbox_inches='tight',pad_inches=0,dpi=300)
        plt.close()

        maes.update(abs(gt - pred))
        mses.update(((gt - pred) * (gt - pred)))
    mae = maes.avg
    mse = np.sqrt(mses.avg)

    print(mae, mse)

if __name__ == '__main__':
    main()
