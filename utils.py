import os
import argparse
import torch
from matplotlib import pyplot as plt
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def print_networks(self, verbose):
    """Print the total number of parameters in the network and (if verbose) network architecture

    Parameters:
        verbose (bool) -- if verbose: print the network architecture
    """
    print('---------- Networks initialized -------------')
    for name in self.model_names:
        if isinstance(name, str):
            net = getattr(self, 'net' + name)
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(results,res_dir,file_name,is_prediction=False):
    """
    input: 
    results: results to save
    res_dir: path where results are gonna be saved
    file_name: name of result file
    is_prediciton: if the results are prediction
    """
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    save_dir = os.path.join(res_dir,file_name)
    f = open(save_dir,'w')
    for result in results:
        if is_prediction:
            for r in result:
                f.writelines(str(r.item())+"\n")
        else:
            f.writelines(str(result)+"\n")
    f.close()
    print("Output finished")
def save_his(his,opt):
    acc_his, time_his,loss_his= his['acc'],his['time'],his['loss']
    res_dir = os.path.join(opt.results,opt.name)

    make_fig(res_dir,acc_his,opt.name,'acc')
    make_fig(res_dir,time_his,opt.name,'time')
    make_fig(res_dir,loss_his,opt.name,'loss')
    save_results(loss_his,res_dir,'loss')
    save_results(acc_his,res_dir,'acc')
    save_results(time_his,res_dir,'time')
def make_fig(res_dir,his,name,title):
    res_name = '%s_%s_his.png' % (name,title)
    res_path = os.path.join(res_dir,res_name)
    plt.figure()
    plt.plot(range(len(his)),his)
    plt.title(title)
    plt.savefig(res_path)
    
def save_networks(model, epoch, save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        name = model.module.model_name
        if isinstance(name, str):
            save_filename = '%s_net_%s.pth' % (epoch, name)
            save_path = os.path.join(save_dir, save_filename)

            if len(model.module.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(model.module.cpu().state_dict(), save_path)
                model.module.cuda(model.module.gpu_ids[0])
            else:
                torch.save(model.module.state_dict(), save_path)