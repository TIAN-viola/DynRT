import pickle

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

class loader_label:
    def __init__(self):
        self.name="label"
        self.require=[]

    def prepare(self,input,opt):
        if opt["test_label"]:
            self.label ={
                "train":load_file(opt["data_path"] + "train_labels"),
                "test":load_file(opt["data_path"] + "test_labels"),
                "valid":load_file(opt["data_path"] + "valid_labels")
            }
        else:
            self.label ={
                "train":load_file(opt["data_path"] + "train_labels"),
                "valid":load_file(opt["data_path"] + "valid_labels")
            }

    def get(self,result,mode,index):
        result["label"]=self.label[mode][index]
        # result["index"]=index

    def getlength(self,mode):
        return len(self.label[mode])