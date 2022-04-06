import os 
from matplotlib import pyplot as plt
import numpy as np
def get_list(path):
    res = []
    with open(path,"r") as f :
        for line in f.readlines():
            line = line.strip("\n")
            res.append(line)
    return res

def make_fig(dics,exp_name,mode='acc'):
    """
    input:
    dics: dictionary of results
    exp_name: name of experiments
    mode: mode of analysis
    output:
    analysis picture of results
    """
    plt.figure()
    if mode =='acc':
        plt.title('accuracy')
        plt.xlabel('num')
        plt.ylabel('acc')
        lines = []
        names = []
        for dic in dics:
            name, acc = dic['name'],dic['acc']
            acc = [np.float32(x) for x in acc]
            # if exp_name == 'length':
            #     name = "%s:%s"%(exp_name,name.split('_')[0])
            # else:  
            #     name = "%s:%s"%(exp_name,name.split('_')[-1])
            names.append(name)
            line, = plt.plot(np.arange(0,len(acc),1),acc)
            lines.append(line)
        plt.legend(lines,names,loc = 'lower right',fontsize = 10)
    elif mode =='time':
        plt.title('time')
        # times = []
        names = [] 
        lines = []
        for dic in dics:
            name, time ,time_list= dic['name'],dic['time_total'],dic['time_list']
            # if exp_name == 'length':
            #     name = "%s:%s"%(exp_name,name.split('_')[0])
            # else:  
            #     name = "%s:%s"%(exp_name,name.split('_')[-1])
            names.append(name)
            # times.append(time)
            line, = plt.plot(np.arange(0,len(time_list),1),time_list)
            lines.append(line)
        plt.legend(lines,names,loc = 'upper left',fontsize = 10)
        plt.xlabel('num')
        plt.ylabel('time')
    elif mode =='loss':
        plt.title('loss')
        plt.xlabel('num')
        plt.ylabel('loss')
        lines = []
        names = []
        for dic in dics:
            name, loss = dic['name'],dic['loss']
            loss = [np.float32(x) for x in loss]
            # if exp_name == 'length':
            #     name = "%s:%s"%(exp_name,name.split('_')[0])
            # else:  
            #     name = "%s:%s"%(exp_name,name.split('_')[-1])
            names.append(name)
            line, = plt.plot(np.arange(0,len(loss),1),loss)
            lines.append(line)
        plt.legend(lines,names,loc = 'upper right',fontsize = 10)
    pic_name = '%s_%s.png'%(exp_name,mode)
    # pic_name = os.path.join('results',pic_name)
    plt.savefig(pic_name)
def analysis(dir,exp_name='dropout'):
    """
    input: 
    dir: path to load experiment results
    exp_name: name of experiment

    output: 
    analysis picture of results
    """
    result_path = dir if exp_name is None else os.path.join(dir,exp_name)
    paths = sorted(os.listdir(result_path))
    dics = []
    for path in paths:
        res_dir = os.path.join(result_path,path)
        acc_dir = os.path.join(res_dir, 'acc')
        time_dir = os.path.join(res_dir,'time')
        loss_dir = os.path.join(res_dir,'loss')
        acc = get_list(acc_dir)
        time = get_list(time_dir)
        loss = get_list(loss_dir)
        loss = [float(x) for x in loss]
        time = [float(x) for x in time]
        time_arr = np.array(time)
        time_total = time_arr.sum()
        name = path
        dic = {'name':name,'acc':acc,'time_list':time,'time_total':time_total,'loss':loss}
        dics.append(dic)
    make_fig(dics,exp_name=exp_name,mode='acc')
    make_fig(dics,exp_name=exp_name,mode='time')
    make_fig(dics,exp_name=exp_name,mode='loss')
if __name__ == "__main__":
    result_path = 'results'
    analysis(result_path,exp_name=None)
    # analysis(result_path,exp_name='length')
    # analysis(result_path,exp_name='width')
    