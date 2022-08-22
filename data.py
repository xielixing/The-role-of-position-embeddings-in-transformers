import re
import random

from numpy import indices
import kaldiark

ff = open('/root/xlx/phones.txt')
label_dict = {}
index = 0
for line in ff:
    label_dict[str(line).replace('\n','')] = index
    index += 1

def parse_scp_line(line):
    m = re.match(r'(.+) (.+):(.+)',line)
    key = m.group(1)
    file = m.group(2)
    shift = int(m.group(3))
    return key,file,shift
def load_data(file_label,file_feature,shift_1,shift_2):
    f1 = open('/root/autodl-tmp/'+file_label)
    f1.seek(shift_1)
    labels = f1.readline().split()
    f1.close()
    f2 = open(file_feature,'rb')
    f2.seek(shift_2)
    features = kaldiark.parse_feat_matrix(f2)
    f2.close()
    return labels,features

class WSJ:
    def __init__(self,feature_scp,label_scp,shuffling=False):
        f = open(feature_scp)
        self.feature_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        f = open(label_scp)
        self.label_entries = [parse_scp_line(line) for line in f.readlines()]
        f.close()

        self.indices = list(range(len(self.feature_entries)))

        if shuffling:
            random.seed(42)
            random.shuffle(self.indices)
    def __iter__(self):
        for i in self.indices:
            k_feature,f_feature,s_feature = self.feature_entries[i]
            k_label,f_label,s_label = self.label_entries[i]
            assert k_feature == k_label

            labels,features = load_data(f_label,f_feature,s_label,s_feature)
            for i in range(len(labels)):
                labels[i] = label_dict[labels[i]]
            yield k_feature,features,labels
