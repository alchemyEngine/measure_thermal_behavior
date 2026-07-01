import json
import sys
import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from os.path import splitext, join


def import_mesh(mesh):
    if all([x in mesh.keys() for x in ['mesh_min',
                                       'mesh_max',
                                       'probed_matrix']]):
        mesh_min = mesh['mesh_min']
        mesh_max = mesh['mesh_max']
        probed_matrix = mesh['probed_matrix']

        n_y_points = len(probed_matrix)
        n_x_points = len(probed_matrix[0])

        x_coords = np.round(np.linspace(mesh_min[0],
                                        mesh_max[0],
                                        n_x_points,
                                        float),
                            decimals=1)
        y_coords = np.round(np.linspace(mesh_min[1],
                                        mesh_max[1],
                                        n_y_points,
                                        float),
                            decimals=1)
        mesh_points = np.array(probed_matrix, float)

        return {'x': x_coords, 'y': y_coords, 'mesh': mesh_points}


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(col_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["white", "black"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_mesh(mesh, title='', subtitle=''):
    fig, ax = plt.subplots(figsize=(8,8))

    data = mesh['mesh']
    # absmax = max(abs(data.min()), abs(data.max()))
    absmax = 0.2
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    im, cbar = heatmap(mesh['mesh'], mesh['x'], mesh['y'], ax=ax,
                       cmap="RdBu_r", norm=norm, cbarlabel="Z-Offset")
    texts = annotate_heatmap(im, valfmt="{x:.4f}")
    plt.gca().invert_yaxis()
    plt.suptitle(title, fontsize=18)
    plt.title(subtitle, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return(fig)


def calc_mesh_delta(mesh_ref, mesh_test):
    delta = mesh_test['mesh'] - mesh_ref['mesh']
    return {'x': mesh_ref['x'], 'y': mesh_ref['y'], 'mesh': delta}


def read_results_file(results_fp):
    with open(results_fp, 'r') as f:
        results = json.load(f)
    return(results)


def plot_deflections(delta, title=''):
    fig, ax = plt.subplots(2)

    fig.tight_layout(rect=[0, 0.03, 1, 0.85])

    x_mid = delta['mesh'][int(np.floor(delta['mesh'].shape[0]-1)/2)]
    x_deflect = - x_mid - min(-x_mid)

    y_min = delta['mesh'][:, 0]
    y_min_deflect = - y_min - min(-y_min)

    y_max = delta['mesh'][:, -1]
    y_max_deflect = - y_max - min(-y_max)

    ax[0].plot(delta['x'],
               x_deflect,
               '-',
               color="#6d89bf")

    ax[0].plot(delta['x'],
               np.zeros(delta['x'].shape),
               '--',
               color="#dddddd")

    ax[1].plot(delta['y'],
               y_max_deflect,
               '-',
               color="#c2d1ed")

    ax[1].plot(delta['y'],
               y_min_deflect,
               '-',
               color="#1a376e")

    ax[1].plot(delta['y'],
               np.zeros(delta['y'].shape),
               '--',
               color="#dddddd")

    ax[0].set_title('Imputed deflection: X axis')
    ax[1].set_title('Y axis (dark left light right)')

    ax[0].set_ylim([-0.2, 0.2])
    ax[1].set_ylim([-0.2, 0.2])

    plt.suptitle(title, fontsize=12)

    return(fig)


def plot_deflection_surface(delta, title=''):
    fig, ax = plt.subplots(2)

    fig.tight_layout(rect=[0, 0.03, 1, 0.85])
    for i in reversed(range(delta['mesh'].shape[0])):
        xgrad = cm.get_cmap('Blues_r',
                            delta['mesh'].shape[0] + 2)
        ax[0].plot(delta['x'],
                   delta['mesh'][i],
                   '-',
                   color=xgrad(i),
                   alpha=1)
    
    for i in reversed(range(delta['mesh'].shape[1])):
        ygrad = cm.get_cmap('Blues_r',
                            delta['mesh'].shape[1] + 2)
        ax[1].plot(delta['y'],
                   delta['mesh'][:,i],
                   '-',
                   color=ygrad(i),
                   alpha=0.8)
        

    ax[0].plot(delta['x'],
               np.zeros(delta['x'].shape),
               '--',
               color="#dddddd")

    ax[1].plot(delta['y'],
               np.zeros(delta['y'].shape),
               '--',
               color="#dddddd")

    ax[0].set_title('X axis surface plot')
    ax[1].set_title('Y axis surface plot')

    ax[0].set_ylim([-0.2, 0.2])
    ax[1].set_ylim([-0.2, 0.2])

    plt.suptitle(title, fontsize=12)
    return(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    print(len(args))
    print(args)
    for infile in args:

        results = read_results_file(infile)

        user = results['metadata']['user']['id']
        printer = results['metadata']['user']['printer']
        backers = results['metadata']['user']['backers']
        x_rails = results['metadata']['user']['x_rails']

        subtitle = '{user}: {printer}\n{backers}, {x_rails}'.format(user=user,
                                                                    printer=printer,
                                                                    backers=backers,
                                                                    x_rails=x_rails)
        premesh = True
        try:
            mesh_pre = import_mesh(results['pre_mesh']['mesh'])
        except KeyError:
            premesh = False
        mesh_ref = import_mesh(results['cold_mesh']['mesh'])
        mesh_test = import_mesh(results['hot_mesh']['mesh'])
        delta = calc_mesh_delta(mesh_ref, mesh_test)


        meshplot = plot_mesh(delta,
                             title='Difference',
                             subtitle=subtitle)
        coldplot = plot_mesh(mesh_ref,
                             title='Cold frame mesh',
                             subtitle=subtitle)
        postplot = plot_mesh(mesh_test,
                             title='Hot frame mesh',
                             subtitle=subtitle)
        defplot = plot_deflections(delta,
                                   title=subtitle)
        surfplot = plot_deflection_surface(delta,
                                           title=subtitle)

        plot_basename = splitext(infile)[0]
        meshplot.savefig('.'.join([plot_basename, 'mesh.png']))
        coldplot.savefig('.'.join([plot_basename, 'coldmesh.png']))
        postplot.savefig('.'.join([plot_basename, 'postmesh.png']))
        defplot.savefig('.'.join([plot_basename, 'deflection.png']))
        surfplot.savefig('.'.join([plot_basename, 'surface.png']))

        if premesh:
            preplot = plot_mesh(mesh_pre,
                                title='Cold bed mesh',
                                subtitle=subtitle)
            bed_delta = calc_mesh_delta(mesh_pre, mesh_ref)
            bedplot = meshplot = plot_mesh(bed_delta,
                                           title='Bed heatup diff',
                                           subtitle=subtitle)
            preplot.savefig('.'.join([plot_basename, 'bed_coldmesh.png']))
            bedplot.savefig('.'.join([plot_basename, 'bed_diffmesh.png']))
            
        plt.close('all')
