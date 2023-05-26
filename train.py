import os
import sys
from math import ceil
import requests
import json
import logging
import time
import random
import torch
import os
import numpy
import input
import model
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score, accuracy_score
import pandas as pd
import datetime
import pickle
import tensorboard_logger as tb_logger
from utils import AverageMeter, LogCollector
import torch.nn as nn
import numpy as np


def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret
class onerun:

    def __init__(self,fname):
        json=onerun.OpenJson(fname)
        assert("info" in json)
        self.info=json["info"]
        assert("opt" in json)
        self.opt=json["opt"]
        if "pth.tar" in self.info["test_on_checkpoint"]:
            self.test_basepath(self.info["test_on_checkpoint"])
            self.BuildLogger()
            self.test_Init()
        else:
            self.basepath()
            self.BuildLogger()
            self.Init()
            assert("name" in self.info)
            self.save_config(json)
        self.train_logger = LogCollector()
        tb_logger.configure(self.opt["tb_logger_path"], flush_secs=5)

    def start(self):
        best_valid_f1 = 0.0
        if len(self.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        for epoch in range(1,self.total_epoch+1):
            train=self.train(epoch)
            tb_logger.log_value('pre_train', train["precision_score"], step=epoch)
            tb_logger.log_value('recall_train', train["recall_score"], step=epoch)
            tb_logger.log_value('f1_train', train["f1_score"], step=epoch)
            tb_logger.log_value('acc_train', train["accuracy"], step=epoch)
            tb_logger.log_value('loss_train', train["loss"], step=epoch)
            valid=self.eval("valid")
            tb_logger.log_value('pre_val', valid["precision_score"], step=epoch)
            tb_logger.log_value('recall_val', valid["recall_score"], step=epoch)
            tb_logger.log_value('f1_val', valid["f1_score"], step=epoch)
            tb_logger.log_value('acc_val', valid["accuracy"], step=epoch)
            tb_logger.log_value('loss_val', valid["loss"], step=epoch)
            is_best = valid["f1_score"] >= best_valid_f1
            best_valid_f1 = max(valid["f1_score"], best_valid_f1)
            if  is_best:
                self.log.info("save best model for now, epoch:" + str(epoch))
                if len(self.device_ids) > 1:
                    self.save_best_checkpoint({
                    'epoch': epoch,
                    'model': self.model.module.state_dict(),
                    'best_f1': best_valid_f1,
                    }, prefix=self.opt["model_savepath"])
                else:
                    self.save_best_checkpoint({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'best_f1': best_valid_f1,
                    }, prefix=self.opt["model_savepath"])
            if "test" in self.opt["mode"]:
                if self.opt["dataloader"]["loaders"]["label"]["test_label"]:
                    test = self.eval("test", epoch=epoch)
                    tb_logger.log_value('pre_test', test["precision_score"], step=epoch)
                    tb_logger.log_value('recall_test', test["recall_score"], step=epoch)
                    tb_logger.log_value('f1_test', test["f1_score"], step=epoch)
                    tb_logger.log_value('acc_test', test["accuracy"], step=epoch)
                    tb_logger.log_value('loss_test', test["loss"], step=epoch)
                    if test["f1_score"] > 0.9388:
                        self.log.info("save test_best_model for now, epoch:" + str(epoch))
                    # self.save_pred_result(test["y_pred"])
                else:
                    pred=self.eval_test("test")
                    self.save_pred_result(pred)
                
            if epoch % self.opt['checkpoint_step'] == 0:
                if len(self.device_ids) > 1:
                    self.save_checkpoint({
                    'epoch': epoch,
                    'model': self.model.module.state_dict(),
                    'best_f1': best_valid_f1,
                    }, prefix=self.opt["model_savepath"])
                else:
                    self.save_checkpoint({
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'best_f1': best_valid_f1,
                    }, prefix=self.opt["model_savepath"])

            self.train_logger.tb_log(tb_logger, step=epoch)
    

    def save_config(self, f_json):
        with open(self.opt["exp_path"] + f_json["info"]["name"] + '.json', 'w') as result_file:
            json.dump(f_json, result_file)

    def save_pred_result(self, pred):
        test_name = load_file(self.opt["dataloader"]["loaders"]["text"]["data_path"] + "test_name")
        predictions_db = pd.DataFrame(data={"img":test_name, "pred":pred})
        predictions_db.to_csv(self.opt["exp_path"] + 'answer.txt', index=False, sep='\t', header=False)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar', prefix=''):
        torch.save(state, prefix + str(state['epoch']) + filename)
    
    def save_best_checkpoint(self, state, filename='model_best.pth.tar', prefix=''):
        torch.save(state, prefix + filename)
        
    def adjust_lr(self, optim, decay_r):
        for param_group in optim.param_groups:
            param_group['lr'] *= decay_r
            
    def calculate_parameters(self):
        # list_parameter = []
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in self.model.parameters():
            mulValue = np.prod(param.size())  
            # list_parameter.append(mulValue)
            Total_params += mulValue  
            if param.requires_grad:
                Trainable_params += mulValue  
            else:
                NonTrainable_params += mulValue  
        # print(list_parameter)
        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def train(self,epoch):
        self.model.train()
        self.log.info('Epoch {}/{}'.format(epoch, self.total_epoch))
        running_loss = 0.0
        running_corrects = 0.0

        y_true = []
        y_pred = []

        self.model.zero_grad()
        
        for i, batch in enumerate(self.dataloaders["train"]):
                
            input={}
            for key in batch:
                input[key]=batch[key].to(self.device)
            
            scores, lang_feat, img_feat = self.model(input)

            loss = self.loss(scores, input["label"], lang_feat, img_feat)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            self.model.zero_grad()
            running_loss += loss.item() * input["label"].size(0)
            
            _, preds = scores.data.max(1)
            running_corrects += (preds == input["label"]).sum()
            y_pred.extend(preds.tolist())
            y_true.extend(input["label"].tolist())

            del input, scores, lang_feat, img_feat
        self.model.zero_grad()

        epoch_loss = running_loss / (len(self.dataloaders["train"]) * self.batch_size)
        epoch_acc = accuracy_score(y_true, y_pred)
        conf=confusion_matrix(y_true, y_pred)
        pre = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        self.log.info(conf)


        self.log.info("train : F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}, Accuracy: {:.4f}, Loss: {:.4f}.".format(f1, pre, recall, epoch_acc, epoch_loss))
        return {
            "confusion_matrix":conf.tolist(),
            "f1_score":f1.item(),
            "precision_score":pre.item(),
            "recall_score":recall.item(),
            "loss":epoch_loss,
            "accuracy":epoch_acc
        }

    def eval(self,mode, epoch=None):
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0.0

        y_true = []
        y_pred = []
        scores_list = []

        with torch.no_grad():
            for i, batch in enumerate(self.dataloaders[mode]):
                
                input={}
                for key in batch:
                    input[key]=batch[key].to(self.device)
                scores, lang_feat, img_feat = self.model(input)

                loss = self.loss(scores, input["label"], lang_feat, img_feat)
                
                running_loss += loss.item() * input["label"].size(0)
                
                _, preds = scores.data.max(1)

                running_corrects += (preds == input["label"]).sum()
                y_pred.extend(preds.tolist())
                scores_list.extend(_.tolist())
                y_true.extend(input["label"].tolist())

                del input, scores

        epoch_loss = running_loss / (len(self.dataloaders[mode]) * self.batch_size)

        epoch_acc = accuracy_score(y_true, y_pred)
        
        conf=confusion_matrix(y_true, y_pred)
        pre_macro = precision_score(y_true, y_pred, average="macro")
        recall_macro = recall_score(y_true, y_pred, average="macro")
        f1_macro = f1_score(y_true, y_pred, average="macro")
        pre = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        self.log.info(conf)
        
        # save predict 
        if epoch:
            np.savez(self.predict_save_path[:-4] + str(epoch) +'.npy', y_pred=np.array(y_pred), y_true=np.array(y_true), score=np.array(scores_list))
            np.save(self.predict_save_score_path[:-4] + str(epoch) +'.npy', np.array(scores_list))
        
        if "pth.tar" in self.info["test_on_checkpoint"]:
            print(mode+": F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}, Accuracy: {:.4f}, Loss: {:.4f}.".format(f1, pre, recall, epoch_acc, epoch_loss))
            self.log.info(mode+": F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}, Accuracy: {:.4f}, Loss: {:.4f}.".format(f1, pre, recall, epoch_acc, epoch_loss))
            return y_true, y_pred

        self.log.info(mode+": F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}, Accuracy: {:.4f}, Loss: {:.4f}.".format(f1, pre, recall, epoch_acc, epoch_loss))
        self.log.info(mode+"-macro: F1: {:.4f}, Precision: {:.4f}, Recall : {:.4f}.".format(f1_macro, pre_macro, recall_macro))
        return {
            "confusion_matrix":conf.tolist(),
            "f1_score":f1.item(),
            "precision_score":pre.item(),
            "recall_score":recall.item(),
            "loss":epoch_loss,
            "accuracy":epoch_acc,
            "y_pred":y_pred

        }
    
    

    def eval_test(self,mode):
        self.model.eval()

        y_pred = []
        with torch.no_grad():
            for i, batch in enumerate(self.dataloaders[mode]):
                
                input={}
                for key in batch:
                    input[key]=batch[key].to(self.device)
                    
                scores, lang_feat, img_feat = self.model(input)

                _, preds = scores.data.max(1)
                y_pred.extend(preds.tolist())

                del input, scores

        return y_pred
    
  

    def Init(self):
        self.InitRandom()
        assert("dataloader" in self.opt)
        self.dataloaders, requirements=self.build_loader(self.opt["dataloader"])

        assert("modelopt" in self.opt and "name" in self.opt["modelopt"])
        self.model=model._models[self.opt["modelopt"]["name"]](self.opt["modelopt"], requirements)
        if "pth.tar" in self.info["train_on_checkpoint"]:
            state_dict=torch.load(self.info["train_on_checkpoint"])
            self.model.load_state_dict(state_dict["model"])
        self.log.info("load model " + self.info["train_on_checkpoint"])    
        if "device" not in self.info:
            self.info["device"]="cuda:0"
        self.device_ids = self.info["device"]
        self.device=torch.device("cuda:"+str(self.device_ids[0]))
        self.log.info("Created Model : %s" % json.dumps(self.opt["modelopt"]))
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        assert("optimizeropt" in self.opt)
        self.BuildOptimizer(self.opt["optimizeropt"])
        self.log.info("Created Optimizer : %s" % json.dumps(self.opt["optimizeropt"]))
        assert("lossopt" in self.opt and "name" in self.opt["lossopt"])
        self.loss=model._loss[self.opt["lossopt"]["name"]](self.opt["lossopt"])
        self.log.info("Created Loss : %s" % json.dumps(self.opt["lossopt"]))
        self.model.to(self.device)
        self.log.info("Model To Device : %s" % self.device)
        self.loss.to(self.device)
        self.log.info("loss To Device : %s" % self.device)
        if "total_epoch" not in self.opt:
            self.opt["total_epoch"]=15
        self.total_epoch=self.opt["total_epoch"]


        if "clip" not in self.opt:
            self.opt["clip"]=1
        self.clip=self.opt["clip"]
        self.log.info("Clip: %s" % self.clip)
        self.predict_save_path = self.opt["exp_path"] + 'predict.npz'
        self.predict_save_score_path = self.opt["exp_path"] + self.opt["name"] +'-predict-score.npy'

    def test_Init(self):

        assert("dataloader" in self.opt)
        self.dataloaders, requirements=self.build_test_loader(self.opt["dataloader"])
        assert("modelopt" in self.opt and "name" in self.opt["modelopt"])

        self.model=model._models[self.opt["modelopt"]["name"]](self.opt["modelopt"], requirements)
        
        state_dict=torch.load(self.info["test_on_checkpoint"])
        self.model.load_state_dict(state_dict["model"])
        self.log.info("load model " + self.info["test_on_checkpoint"])    
        if "device" not in self.info:
            self.info["device"]="cuda:0"
        self.device_ids = self.info["device"]
        self.device=torch.device("cuda:"+str(self.device_ids[0]))
        self.log.info("Created Model : %s" % json.dumps(self.opt["modelopt"]))
        self.model.to(self.device)

        self.log.info("Model To Device : %s" % self.device)
        self.loss=model._loss[self.opt["lossopt"]["name"]](self.opt["lossopt"])
        self.loss.to(self.device)
        self.log.info("loss To Device : %s" % self.device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        

        self.predict_save_path = self.opt["exp_path"] + 'predict.npz'
        self.predict_save_score_path = self.opt["exp_path"] + self.opt["name"] +'-predict-score.npy'

    def BuildOptimizer(self,opt):
        assert("name" in opt)
        dic={}
        for n in self.model.named_modules():
            dic[n[0]]=n[1]
        assert("lr" in opt)
        lr=opt["lr"]
        if "weight_decay" not in opt:
            opt["weight_decay"]=0
        weight_decay=opt["weight_decay"]
        assert("params" in opt)
        params=[]
        self.training=0
        for p in opt["params"]:
            popt=opt["params"][p]
            ppopt={'params': dic[p].parameters()}
            if "lr" in popt:
                ppopt["lr"]=popt["lr"]
            if "pweight_decay" in popt:
                ppopt["pweight_decay"]=popt["pweight_decay"]
            params.append(ppopt)
            for p in dic[p].parameters():
                if p.requires_grad:
                    self.training+=p.numel()

        self.optimizer=model._optimizers[opt["name"]](params=params,lr=lr,weight_decay=weight_decay)


    def InitRandom(self):
        if "seed" not in self.opt:
            self.opt["seed"]=str(ceil(time.time()))
        seed=int(self.opt["seed"])
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED']=str(seed)
        self.log.info("Set Seed : %s" % seed)
    
    def build_loader(self,opt):
        loaders={}
        requireopt=opt["requires"]
        self.log.info("Require : %s" % list(requireopt.keys()))

        inputs={}
        for key in requireopt:
            input._requires[key](inputs,requireopt[key])
            self.log.info("Loaded %s : %s" % (key,json.dumps(requireopt[key])))
        loaderopt=opt["loaders"]
        for key in loaderopt:
            loaders[key]=input._loadermap[key]
            loaders[key].prepare(inputs,loaderopt[key])
            self.log.info("Prepared %s : %s" % (key,json.dumps(loaderopt[key])))
        
        self.batch_size=opt["batch_size"] if "batch_size" in opt else 16
        pin_memory=opt["pin_memory"] if "pin_memory" in opt else True
        num_workers=opt["num_workers"] if "num_workers" in opt else 8
        shuffle=opt["shuffle"] if "shuffle" in opt else True
        dataloaders={}

        
        dataloaders["train"] = torch.utils.data.DataLoader(
            Dataset(loaders,"train"),
            batch_size=self.batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=True
        )
        dataloaders["valid"] = torch.utils.data.DataLoader(
            Dataset(loaders,"valid"),
            batch_size=self.batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=False
        )

        if "test" in self.opt["mode"]:
            dataloaders['test'] = torch.utils.data.DataLoader(
                Dataset(loaders,'test'),
                batch_size=self.batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False
            )
        self.log.info("Created DataLoaders : %s" % json.dumps({
                "batch_size":self.batch_size,
                "pin_memory":pin_memory,
                "num_workers":num_workers,
                "shuffle":shuffle
        }))
        return dataloaders, inputs

    def build_test_loader(self,opt):
        loaders={}
        requireopt=opt["requires"]
        self.log.info("Require : %s" % list(requireopt.keys()))

        inputs={}
        for key in requireopt:
            input._requires[key](inputs,requireopt[key])
            self.log.info("Loaded %s : %s" % (key,json.dumps(requireopt[key])))
        loaderopt=opt["loaders"]
        for key in loaderopt:
            loaders[key]=input._loadermap[key]
            loaders[key].prepare(inputs,loaderopt[key])
            self.log.info("Prepared %s : %s" % (key,json.dumps(loaderopt[key])))
        
        self.batch_size=opt["batch_size"] if "batch_size" in opt else 16
        pin_memory=opt["pin_memory"] if "pin_memory" in opt else True
        num_workers=opt["num_workers"] if "num_workers" in opt else 8
        shuffle=opt["shuffle"] if "shuffle" in opt else True
        drop_last=opt["drop_last"] if "drop_last" in opt else True
        dataloaders={}

        dataloaders["train"] = torch.utils.data.DataLoader(
            Dataset(loaders,"train"),
            batch_size=self.batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )
        dataloaders["valid"] = torch.utils.data.DataLoader(
            Dataset(loaders,"valid"),
            batch_size=self.batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )
        
        dataloaders['test'] = torch.utils.data.DataLoader(
            Dataset(loaders,'test'),
            batch_size=self.batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
            drop_last=False
        )
        self.log.info("Created DataLoaders : %s" % json.dumps({
                "batch_size":self.batch_size,
                "pin_memory":pin_memory,
                "num_workers":num_workers,
                "shuffle":shuffle,
                "drop_last":drop_last
        }))
        return dataloaders, inputs


    def basepath(self):
        tick=datetime.datetime.now().strftime("%m-%d-%H_%M_%S")
        self.opt["name"]=tick
        self.opt["exp_path"] = './exp/' + self.opt["name"] + '/'
        self.opt["model_savepath"] = 'exp/' + self.opt["name"] + '/' + 'checkpoints/'
        self.opt["tb_logger_path"] = 'exp/' + self.opt["name"] + '/' + 'runs/'
        if not os.path.exists('exp/'):
            os.makedirs('exp/')
        os.makedirs(self.opt["exp_path"])
        os.makedirs(self.opt["model_savepath"])
        os.makedirs(self.opt["tb_logger_path"])

    
    def test_basepath(self, model_path):
        tick=model_path.split('/')[1]
        self.opt["name"]=tick
        self.opt["exp_path"] = './exp/' + self.opt["name"] + '/'
        self.opt["model_savepath"] = 'exp/' + self.opt["name"] + '/' + 'checkpoints/'
        self.opt["tb_logger_path"] = 'exp/' + self.opt["name"] + '/' + 'runs/'


    def BuildLogger(self):
        assert("log" in self.info)
        opt=self.info["log"]
        level=logging._nameToLevel[opt["level"]] if "level" in opt else logging.INFO
        format=opt["format"] if "format" in opt else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.log=logging.getLogger(__name__)
        self.log.setLevel(level=level)
        handler=logging.FileHandler("./exp/"+ self.opt["name"] +'/' + 'log.txt')
        handler.setLevel(level=level)
        handler.setFormatter(logging.Formatter(format))
        self.log.addHandler(handler)
        console=logging.StreamHandler()
        console.setLevel(level=level)
        console.setFormatter(logging.Formatter(format))
        self.log.addHandler(console)
        self.log.info("start logging : %s" % json.dumps({
            "fname":"./exp/"+ self.opt["name"] +'/' + 'log.txt',
            "level":logging._levelToName[level],
            "format":format
        }))
        self.log.info("info: %s" % json.dumps(self.info))

    def OpenJson(fname):
        with open(fname,"r") as f:
            return json.load(f)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, loaders, mode, test_label=True):
        if test_label:
            self.loaders=loaders
        else:
            loaders_ = loaders.copy()
            del loaders_['label']
            self.loaders = loaders_       
        self.mode=mode
        self.length=0
        count=0
        for loader in self.loaders.values(): # check lens fo values of loaders whether are same
            l=loader.getlength(mode)
            if self.length!=l:
                self.length=l
                count+=1
        assert(count<2)

    def __getitem__(self, index):
        output={}
        for key in self.loaders:
            self.loaders[key].get(output,self.mode,index)
        return output
    
    def __len__(self):
        return self.length



def main(args):
    print(args)
    # assert(len(args)==2)
    fname=args[1]
    # fname = 'config/DynRT.json'
    assert(os.path.exists(fname) and fname.endswith(".json"))
    OneRun=onerun(fname)
    OneRun.start()

    

if __name__=="__main__":
    main(sys.argv)