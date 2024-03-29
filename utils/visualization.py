import mayavi.mlab as mlab


def plot_pc_mayavi(pc_lst, colors, show=True):
    mlab.figure()
    for pc, color in zip(pc_lst, colors):
        mlab.points3d(*pc.T, scale_factor=.025, color=color)
    if show:
        mlab.show()
