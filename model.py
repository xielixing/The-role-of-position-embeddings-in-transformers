import torch
from torch import nn
import torch.nn.functional as F
import data
import math
from torch.autograd import Variable

class SelfAttention(nn.Module):
    def __init__(self, emb, heads):
        super().__init__()
        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads

        # - We will break the embedding into `heads` chunks and feed each to a different attention head
        self.tokeys    = nn.Linear(emb, emb)
        self.toqueries = nn.Linear(emb, emb)
        self.tovalues  = nn.Linear(emb, emb)
        self.toout     = nn.Linear(emb,emb)
        self.unifyheads = nn.Linear(emb,emb)

   
    def forward(self,x):
        Q = self.toqueries(x).reshape(-1,x.shape[0],x.shape[1],self.emb//self.heads)
        K = self.toqueries(x).reshape(-1,x.shape[0],x.shape[1],self.emb//self.heads)
        V = self.toqueries(x).reshape(-1,x.shape[0],x.shape[1],self.emb//self.heads)
        attention_score = torch.matmul(Q,K.permute(0,1,3,2))*1/math.sqrt(self.emb)
        output = torch.matmul(F.softmax(attention_score,dim=-1),V).reshape(x.shape[0],x.shape[1],-1)
        return self.unifyheads(output)

class TransformerBlock(nn.Module):
    
    def __init__(self, emb, heads,ffn=4*256):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)
        self.drop = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(emb,ffn)
        self.linear2 = nn.Linear(ffn,emb)
        self.relu = nn.ReLU()


    def forward(self, x):
        attended = self.attention(x)
        #attended = self.drop(attended)
        x = self.norm1(attended + x)
        #x = self.drop(x)
        ff = self.linear1(x)
        ff = self.relu(ff)
        #ff = self.drop(ff)
        ff = self.linear2(ff)
        #ff = self.drop(ff)
        x = self.norm2(ff + x)
        return x

class CTransformer(nn.Module):
    def __init__(self, emb, heads, depth, num_classes,max_len=3000):
        super().__init__()
        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs1 = nn.Linear(emb,256) # 分类头
        self.toprobs2 =nn.Linear(256,num_classes)
        self.drop = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.toemb = nn.Linear(40,emb)

    def forward(self, x):
        x = self.toemb(x)
        x = x.view(1,x.size(0),x.size(1)) # 1 652 43
        #x = torch.tensor(x,dtype=torch.float)
        x = self.tblocks(x)
        # x = self.toprobs(x.squeeze(0))
        x = self.toprobs1(x.squeeze(0))
        #x = self.drop(x)
        x = self.relu(x)
        x = self.toprobs2(x)
        return x


def weight_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight)
        nn.init.constant_(model.bias, 0)

def calculate_accuracy(out,real):
    pred = torch.argmax(out,dim=1)
    true_value = real
    pred_true = 0
    correct_cmp = pred.type(true_value.dtype) == true_value
    pred_true = correct_cmp.sum()
    return pred_true

def evaluate(model):
    model.eval()
    total_dev_samples = 0
    total_dev_right_predictions = 0
    total_dev_loss = 0
    m = '/root/xlx/si284-0.9-dev.fbank.scp'
    n = '/root/xlx/si284-0.9-dev.bpali.scp'
    dataset = data.WSJ(m,n)
    i = 0
    for key,feature,labels in dataset:
        i += 1
        mean = feature.mean(-1,keepdims=True)
        std = feature.std(-1, keepdims=True)
        feature = (feature - mean) / (std+1e-12)
        dev_input = torch.Tensor(feature).cuda()
        #dev_input = dev_input + pe[:dev_input.size(0), :dev_input.size(1)].cuda()
        labels = torch.tensor(labels).cuda()
        with torch.no_grad():
            dev_out = model(dev_input)
        total_dev_right_predictions += calculate_accuracy(dev_out,labels)
        total_dev_samples += dev_input.shape[0]
        loss = nn.CrossEntropyLoss()
        total_dev_loss += loss(dev_out,labels).item()
    acc_epoch.append((total_dev_right_predictions/total_dev_samples).item())
    val_loss.append(total_dev_loss/i)
    print('dev accuracy:\t',total_dev_right_predictions/total_dev_samples)
    print('dev loss\t',total_dev_loss/i)

if __name__ == "__main__":

# shuffle
# chaneg the area of pe/normalize
# change parameters
# attention score
    num_epochs = 10
    lr = 0.0003
    embedding_size = 256
    num_heads = 8
    depth = 6
    order = 'Normal'
    loss_epoch = []
    acc_epoch = []
    val_loss = []
    hidden_units = 1024 #512
    max_len = 3000

    pe = torch.zeros(max_len,embedding_size).cuda()
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_size, 2) *
                             -(math.log(10000.0) / embedding_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    #create the model
    model = CTransformer(emb=embedding_size,heads=num_heads,depth=depth,num_classes=42).cuda()
    model_loaded = CTransformer(emb=embedding_size,heads=num_heads,depth=depth,num_classes=42).cuda()
    #model.apply(weight_init)
    model.load_state_dict(torch.load('/root/xlx/models/14/epoch_21.pth')['model'])
    opt = torch.optim.Adam(lr=lr,params=model.parameters())
    #opt = torch.optim.SGD(lr=0.05,params=model.parameters())
    # sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=2,
    #                                             verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    #sch = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
    feature_scp = '/root/xlx/si284-0.9-train.fbank.scp'
    label_scp = '/root/xlx/si284-0.9-train.bpali.scp'
    dataset = data.WSJ(feature_scp,label_scp)

    #training loop
    for e in range(num_epochs):
        print(f'\n epoch {e}')
        model.train(True)
        #model.eval()
        train_loss = 0
        index = 0
        for key,feature,labels in dataset:
            mean = feature.mean(-1,keepdims=True)
            std = feature.std(-1, keepdims=True)
            feature = (feature - mean) / (std+1e-12)
            input_tensor = torch.Tensor(feature).cuda()
            input_tensor = input_tensor + pe[:input_tensor.size(0), :input_tensor.size(1)].cuda()
            label_tensor = torch.LongTensor(labels).cuda()
            opt.zero_grad()
            out_tensor = model(input_tensor)
            loss = nn.CrossEntropyLoss()
            crossEntropyLoss = loss(out_tensor,label_tensor)
            train_loss += crossEntropyLoss.item()
            crossEntropyLoss.backward()
            opt.step()
            index += 1
            if index % 10000 == 0:
                print('training loss:\t',train_loss/index)
            
        #sch.step()
        loss_mean = train_loss / (index)
        loss_epoch.append(loss_mean)
        state = {'model':model.state_dict()
            ,'optimizer':opt.state_dict()
            ,'num_epochs':num_epochs
            ,'lr':lr
            ,'embedding_size':embedding_size
            ,'num_heads':num_heads
            ,'depth':depth
            ,'order':order
            ,'loss_epoch':loss_epoch
            ,'hiden_units':hidden_units
            ,'acc_epoch':acc_epoch
            ,'val_loss':val_loss
            }
        torch.save(state,'/root/xlx/models/14/'+'epoch_'+str(e+22)+'.pth')
        checkpoint = torch.load('/root/xlx/models/14/'+'epoch_'+str(e+22)+'.pth')
        model_loaded.load_state_dict(checkpoint['model'])
        evaluate(model_loaded)
        print('Train Epoch: {}\t Loss: {:.6f}'.format(e,loss_mean))
print('training completed !')
