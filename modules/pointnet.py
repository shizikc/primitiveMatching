import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.in1 = nn.InstanceNorm1d(64)
        self.in2 = nn.InstanceNorm1d(128)
        self.in3 = nn.InstanceNorm1d(1024)
        self.in4 = nn.InstanceNorm1d(512)
        self.in5 = nn.InstanceNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]

        if batchsize > 1:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.in1(self.conv1(x)))
            x = F.relu(self.in2(self.conv2(x)))
            x = F.relu(self.in3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = self.fc1(x)
            x = self.fc2(x)
            #
            # x = F.relu(self.in4(x))
            # x = F.relu(self.in5(self.fc2(x)))

        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.in1 = nn.InstanceNorm1d(64)
        self.in2 = nn.InstanceNorm1d(128)
        self.in3 = nn.InstanceNorm1d(1024)
        self.in4 = nn.InstanceNorm1d(512)
        self.in5 = nn.InstanceNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        if batchsize > 1:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)
        else:
            x = F.relu(self.in1(self.conv1(x)))
            x = F.relu(self.in2(self.conv2(x)))
            x = F.relu(self.in3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            x = F.relu(self.in4(self.fc1(x)))
            x = F.relu(self.in5(self.fc2(x)))
            x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.in1 = nn.InstanceNorm1d(64)
        self.in2 = nn.InstanceNorm1d(128)
        self.in3 = nn.InstanceNorm1d(1024)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        bs = x.shape[0]

        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        if bs > 1:
            x = F.relu(self.bn1(self.conv1(x)))
        else:
            x = F.relu(self.in1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        if bs > 1:
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.in2(self.conv2(x)))
            x = self.in3(self.conv3(x))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)

        self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, k)

        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, k, bias=True)

        self.dropout = nn.Dropout(p=0.3)

        self.in1 = nn.InstanceNorm1d(512)
        self.in2 = nn.InstanceNorm1d(512)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(2, 1)

        x, _, _ = self.feat(x)  # torch.Size([bs, 1024])
        if x.shape[0] > 1:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        else:
            x = self.fc1(x)
            print(x.shape)
            x = F.relu(self.in1(x))
            x = F.relu(self.in2(self.dropout(self.fc2(x))))
        x = torch.sigmoid(self.fc3(x))
        return x
        # return F.log_softmax(x, dim=1)


class PointNetDenseCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.in1 = nn.InstanceNorm1d(512)
        self.in2 = nn.InstanceNorm1d(256)
        self.in3 = nn.InstanceNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x, _, _ = self.feat(x)  # torch.Size([bs, 1088, num_points])
        if x.shape[0] > 1:
            x = F.relu(self.bn1(self.conv1(x)))  # : torch.Size([bs, 512, num_points])
            x = F.relu(self.bn2(self.conv2(x)))  # torch.Size([bs, 256, num_points])
            x = F.relu(self.bn3(self.conv3(x)))  # torch.Size([bs, 128, num_points])
        else:
            x = F.relu(self.in1(self.conv1(x)))  # : torch.Size([bs, 512, num_points])
            x = F.relu(self.in2(self.conv2(x)))  # torch.Size([bs, 256, num_points])
            x = F.relu(self.in3(self.conv3(x)))  # torch.Size([bs, 128, num_points])
        x = self.conv4(x)  # torch.Size([bs, k, points])
        # x = x.transpose(2, 1).contiguous()  # torch.Size([bs, num_points, k])
        # x = x.contiguous()  # torch.Size([bs, k, points])

        return x


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(2, 2500, 3))
    trans = STN3d()
    #

    cls = PointNetCls(k=100)
    out = cls(sim_data)
    print('class', out.size())  # torch.Size([bs, k]).

    seg = PointNetDenseCls(k=50)
    out = seg(sim_data)
    print('seg', out.size())  # torch.Size([bs, k, num_points])
