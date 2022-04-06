from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import os
from utils import *
from torch.nn import init
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import importlib
import functools
from models.networks import *
import time
from tensorboardX import SummaryWriter
import tqdm
# 数据加载
class CifarDataset(Dataset):
    def __init__(self,opt,phase):
        """
        input:
        opt: hyperparameters
        phase: experient phase
        
        output:
        
        """
        # 这里添加数据集的初始化内容
        self.dir = opt.dataroot
        self.img_root = os.path.join(self.dir,"image")
        self.file_dir = None
        self.img_path = []
        self.labels = []
        name = '%sset.txt'%phase
        self.file_dir = os.path.join(self.dir,name)
        with open(self.file_dir,"r") as f :
            for line in f.readlines():
                line = line.strip("\n").split(' ')
                if len(line)>1:
                    self.labels.append(line[1])
                self.img_path.append(line[0])

        self.size  = len(self.img_path)
        self.transform = self.get_transform()

    def get_transform(self):
        """
        input:None
        output: transforms 
        """
        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        # transform_list.append(transforms.RandomGrayscale())
        # transform_list.append(transforms.RandomHorizontalFlip())
        # transform_list.append(transforms.RandomCrop(32, padding=4))
        # transform_list.append(transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)))
        # transform_list.append(transforms.Resize((224,224)))
        return transforms.Compose(transform_list)
    def __getitem__(self, index):
        """
        input:
        index: index of input image
        output: 
        dic: a dictionary of image and its label or a dictionary of image
        """
        path = os.path.join(self.img_root,self.img_path[index % self.size])  # make sure index is within then range
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        if len(self.labels)>0:
            label = int(self.labels[index % self.size])
            return {'img': img, 'label': label}
        # apply image transformation
        return {'img': img}
    def __len__(self):
        # 这里添加len函数的相关内容
        """
        input: None
        output: length of Dataset
        """
        return self.size

# 定义 train 函数
def train():
    # 参数设置
    epoch_num = opt.n_epochs+opt.n_epochs_decay
    val_num = 2
    total_iters = 0
    acc_his = []
    time_his = []
    loss_his = []
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    log_dir = os.path.join(opt.logs_dir,opt.name)
    # writer = SummaryWriter(log_dir)
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        update_learning_rate(optimizer,schedulers)
        epoch_start_time = time.time()  # timer for entire epoch
        total_step = len(train_loader)
        l = 0 #average loss for each epoch
        for index, data in enumerate(train_loader, 0):
            total_iters += opt.batch_size
            iter_data_time = time.time()
            images, labels = data['img'],data['label']
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward
            output = net(images)
            # Backward
            loss = criterion(output, labels)
            loss.backward()
            # Update
            optimizer.step()
            # writer.add_scalar("cross_entropy",loss.item(),global_step=total_iters)
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time:{}' 
                   .format(epoch+1, epoch_num, index+1, total_step, loss.item(), time.time()-iter_data_time))
            
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch_num, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                save_networks(net,save_suffix,save_dir)
            l += loss.item()
            # 模型训练n轮之后进行验证
        if epoch % val_num == 0:
            acc_his.append(validation())
        epoch_time = time.time()-epoch_start_time
        time_his.append(epoch_time)
        loss_his.append(l/total_step)
        print("End of epoch{},Use time:{}".format(epoch_num+1,epoch_time))
        print("Saving the latest model....")
        save_networks(net,epoch='latest',save_dir=save_dir)
    print('Finished Training!')
    his = {'acc':acc_his,'time':time_his,'loss':loss_his}
    save_his(his,opt)

# 定义 validation 函数
def validation():
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in dev_loader:
            images, true_labels = data['img'],data['label']
            # 在这一部分撰写验证的内容，下面两行不必保留
            images = images.to(device)
            labels = true_labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)  ##更新测试图片的数量   size(0),返回行数
            correct += (predicted == labels).sum().item() ##更新正确分类的图片的数量
    accuracy = correct/total
    print("验证集数据总量：", total, "预测正确的数量：", correct)
    print("当前模型在验证集上的准确率为：", accuracy)
    return accuracy


# 定义 test 函数
def test():
    """
    input:data_loader for test
    output: a txt file of prediction
    """
    # 测试函数，需要完成的任务有：根据测试数据集中的数据，逐个对其进行预测，生成预测值。
    # 将结果按顺序写入txt文件中，下面一行不必保留
    res_dir = os.path.join(opt.results,opt.name)
    model =  create_model(opt)
    model_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model_path = os.path.join(model_dir, 'latest_net_%s.pth'%opt.model)
    model.module.load_state_dict(torch.load(model_path)) #load model
    results = []
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
        for data in tqdm.tqdm(test_loader):
            images = data['img']
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) 
            results.append(predicted)
    
    save_results(results,res_dir,"prediction",is_prediction=True)

def find_model_using_name(model_name):
    """Import the module "models/[model_name].py".
    """
    if "resnet" in model_name:
        name = "resnet"
    elif 'vgg' in model_name:
        name = "vgg"
        model_name = name
    else:
        name = model_name
    model_filename = "models."+name
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '')
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model
def create_model(opt):
    """Create a model given the option.
    """

    model = find_model_using_name(opt.model)
    instance = define_network(opt,model)
    print("model [%s] was created" % type(instance.module).__name__)
    return instance
"""parsing and configuration"""

def parse_args():
    desc = "Pytorch implementation of FC network"
    parser = argparse.ArgumentParser(description=desc)
    #preparing
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset_name')
    parser.add_argument('--name', type=str, default='resnet18_noenhance', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--dataroot', type=str, default='Dataset', help='path of dataset')
    parser.add_argument('--model', type=str, default='resnet18', help='chooses which model to use. [lenet|resnet18|resnet34|resnet50|vgg...]')
    parser.add_argument('--results',type=str, default='./results',help='results are saved here')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=10, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--logs_dir', type=str, default='./logs', help='logs for tensorboardX are saved here')
    parser.add_argument('--gpu_ids', type=str, default='2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    #training
    parser.add_argument('--print_freq', type=int, default=500, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=500000, help='frequency of saving the latest results')
    parser.add_argument('--batch_size', type=int, default=20, help='The size of batch size')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--epoch', type=int, default=20, help='The number of epoch')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=40, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
    parser.add_argument('--num_val', type=int, default=5, help='start valication when epoch reach ...')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='float("inf"),Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    return parser.parse_args()
if __name__ == "__main__":
    opt = parse_args()
    opt.save_by_iter = True
    if opt is None:
      exit()
    #make checkpoints dir 
    expr_dir = [os.path.join(opt.checkpoints_dir, opt.name),os.path.join(opt.results, opt.name),os.path.join(opt.logs_dir,opt.name)]
    mkdirs(expr_dir)
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    # 构建数据集
    train_set = CifarDataset(opt,"train")
    dev_set = CifarDataset(opt,"valid")
    test_set = CifarDataset(opt,'test')

    # 构建数据加载器
    train_loader = DataLoader(
        dataset=train_set,
        batch_size= opt.batch_size,
        shuffle=True
    )
    dev_loader = DataLoader(
        dataset=dev_set,
        batch_size= opt.batch_size,
        shuffle=True
        )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size= opt.batch_size,
        shuffle=False
    
    )
    # 初始化模型对象
    net = create_model(opt)
    # net = Net(opt)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
    # net = net.to(device)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))  # torch.optim中的优化器进行挑选，并进行参数设置
    schedulers = get_scheduler(optimizer, opt)
    # 模型训练
    train()

    # 对模型进行测试，并生成预测结果
    # test()
