import matplotlib
import matplotlib.pyplot as plt
import numpy as np

W = 10
H = 0.4 * W

matplotlib.rc('text')
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False

FONT_SIZE = 15
MARKER_SIZE = 15
LINEWIDTH = 2
SAMPLE_SIZE = 3

plt.rc('font', size=FONT_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title

def swarmplot(margins, alpha=0.3, jitter=0.1, color='red', s=SAMPLE_SIZE, discard_ratio=0):
    plt.figure(figsize=(W, H))
    idx = np.tile(np.arange(margins.shape[0]), (margins.shape[1], 1)).T
    plt.plot(idx[:,0], np.median(margins, axis=1), '.-', linewidth=LINEWIDTH, color=color, markersize=MARKER_SIZE)
    idx = idx[:, :int(margins.shape[1] * (1-discard_ratio))]
    max_median = np.median(margins, axis=1).max()
    margins = margins[:,:int(margins.shape[1] * (1-discard_ratio))]
    plt.scatter(idx[:]+ jitter * np.random.randn(*margins.shape), margins[:], alpha=alpha, color=color, s=s)
    
    new_xticks=['Low', 'Frequency', 'High']
    plt.xticks([-1, 10, 21], new_xticks, rotation=0, horizontalalignment='center')
    plt.axis([-1, 21, 0, max_median * 1.3])
    plt.ylabel('Margin')
