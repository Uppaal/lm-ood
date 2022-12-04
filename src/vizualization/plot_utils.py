from setup import paths
import matplotlib.pyplot as plt


def plot_historgram(scores, labels, title, x_label='Score', y_label='Frequency', savename=None):
    """

    :param scores: List of list of scores to plot. Each sublist represents one histogram.
    :param labels: List of labels for each histogram. Same length as scores.
    :param title: Title of plot
    :param x_label: Name for x-axis
    :param y_label: Name for y-axis
    :param savename: Filename to save plot with. If None, nothing is saved.
    :return:
    """

    plt.clf()
    for i in range(len(scores)):
        plt.hist(scores[i], label=labels[i], alpha=0.3, bins='auto')

    plt.legend()
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if savename is not None:
        plt.savefig(f'{paths.output_plot_dir}/{savename}')
    plt.show()


def plot_bar(x, y, x_label, y_label, title, savename=None, colours_list=None, colours_dict=None):
    """
    Plots a bar graph of two lists.
    :param x: List of values on the x-axis
    :param y: List of values on the y-axis, i.e. height
    :param x_label: Name for x-axis
    :param y_label: Name for y-axis
    :param title: Title of plot
    :param savename: Filename to save plot with. If None, nothing is saved.
    :param colours_list: List of str colours for each bar on the x-axis.
    :param colours_dict: E.g. {'ID': 'red', 'OOD': 'blue'}
    :return:
    """

    plt.clf()
    plt.bar(x=x, height=y,
            color=colours_list,
            tick_label=range(len(x)))

    if colours_list is not None and colours_dict is not None:
        labels = list(colours_dict.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colours_dict[label]) for label in labels]
        plt.legend(handles, labels)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if savename is not None:
        plt.savefig(f'{paths.output_plot_dir}/{savename}')
    plt.show()


def plot_lines(x, y_list, labels, x_label, y_label, title, savename=None, colours_list=None, y_bins=None):
    """
    Plots a set of lines onto the same graph.
    :param x: List of values on the x-axis. Can be numerical (int, float) or str.
    :param y_list: List of lists. Each sublist are the y values for one line graph.
    :param labels: List of str labels for each line graph.
    :param x_label: Title for x-axis.
    :param y_label: Title for y-axis.
    :param title: Title for plot.
    :param savename: If not None, saves plot with this filename.
    :param colours_list: Optional. List of colours for each line graph.
    :param y_bins: Optional. Number of ticks to show on the y-axis.
    :return:
    """
    plt.clf()

    if colours_list is not None:
        for i, y in enumerate(y_list):
            plt.plot(x, y, label=labels[i], color=colours_list[i])
    else:
        for i, y in enumerate(y_list):
            plt.plot(x, y, label=labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid()

    if y_bins is not None:
        plt.locator_params(axis='y', nbins=5)

    if savename is not None:
        plt.savefig(f'{paths.output_plot_dir}/{savename}')
    plt.show()


def string_to_list(str, convert_to_float=True):
    l = str.split()
    if convert_to_float:
        return [float(x) for x in l]
    return l


k = string_to_list("""1
10
50
100
200
300
400
500
600
1k
2k
5k
10k""", convert_to_float=False)

# plot_lines(x=k,
#            y_list=s1, s2, s3, s4,
#            labels = ['FT (kNN)', 'FT (k-avg NN)', 'FT+TAPT (kNN)', 'FT+TAPT (k-avg NN)'],
#            x_label='k', y_label='Score',
#            title='FPR@95 (20NG -> SST-2)',
#            savename='effect-of-k_20NG-to-SST2_FPR.png',
#            colours_list=None, y_bins=5)
#
# print('hi')
