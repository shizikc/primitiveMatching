import sys
import h5py

from utils.visualization import plot_pc_mayavi

if __name__ == '__main__':
    path = "C:\\Users\\sharon\\Documents\\Research\\data\\dyna\\dyna_dataset_f.h5"
    # path = "/home/coopers/data/dyna/dyna_dataset_f.h5"
    sidseq = "50025_shake_shoulders"
    f = h5py.File(path, 'r')  # as f:
    kys = list(f.keys())
    # if sidseq not in f:
    #     # print('Sequence %s from subject %s not in %s' % (seq, sid, path))
    #     f.close()
    #     sys.exit(1)
    # for sidseq in kys:
    # print(sidseq)
    verts = f[sidseq].value.transpose([2, 0, 1])
    faces = f['faces'].value

    plot_pc_mayavi([verts[1]], colors=[(0.5, 0.5, 0.5)])
