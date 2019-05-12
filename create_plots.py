import matplotlib.pyplot as plt
import glob, os
import pickle
import pandas as pd
from decimal import Decimal
import numpy as np
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')


def plot_data(df,count,legend):

    # Setup Graphs
    node_string = ''
    i = 1
    hidden_nodes = df['node_count'].iloc[0]
    VC = df['VCdim'].iloc[0]
    W = df['param_count'].iloc[0]
    L = df['depth'].iloc[0]
    while i < L:
        node_string = node_string + r'$h_%s$ = ' % i + str(hidden_nodes) + ', '
        i += 1

    # Build Sub-Graphs
    key_str = "$VC_{max}$: upper bound of VC dimension \n$W$: total # of network parameters \n $h_i$: number of nodes in hidden layer $i$"
    ax = plt.subplot(2, 1, 1)
    # cdict = ['red', 'blue', 'green', 'purple', 'orange','pink','brown']
    cdict = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'RdPu', 'Copper']
    colors = np.r_[np.linspace(0.2, .7, 3), np.linspace(0.2, .7, 3)]
    mymap = plt.get_cmap(cdict[count])
    # get the colors from the color map
    my_colors = mymap(colors)
    plt.scatter(df['epoch'],df['loss'],s=.7, color=my_colors[2],label=(
                r'$\bf VC_{max}$ = ' + r'%.2E' % Decimal(str(VC)) + r'   $\bf W$ = ' + '%.2E' % Decimal(
            str(W)) + '   ' + node_string ))
    # plt.title('Network Depth = '+str(L),fontsize=15,loc='left')
    if legend:
        plt.legend(loc='upper right', markerscale=5, bbox_to_anchor=(1, 1.35),
                   ncol=1, fancybox=True, shadow=True, fontsize=9,
                   title=r'$\bf Network Parameters$' + '\n          (depth=' + str(L) + ')',prop=fontP)


    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    plt.text(0, 1.25, key_str, fontsize=10,
             horizontalalignment='left',
             verticalalignment='top', bbox=props, transform=ax.transAxes)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(color='gray', linestyle='--', linewidth=.5)
    plt.subplot(2, 1, 2)
    plt.scatter(df['epoch'],df['binary_accuracy'],s=.7,color=my_colors[2])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(color='gray', linestyle='--', linewidth=.5)


def customWeightBatch(batch,count,legend):
    batch['cL'] = batch['node_count'].apply(lambda x: np.where(x == np.max(x))[0][0] + 1)
    batch['custom_weights'] = batch['node_count'].apply(lambda x: not isinstance(x, int))

    # Strip out any runs that were not using custom_weights
    batch = batch[batch['custom_weights']]

    layers = batch['depth'].unique()
    num_lay = len(layers)
    cLayers = batch['cL'].unique()
    num_cls = len(cLayers)
    # params = batch['param_count'].unique()
    # num_params = len(params)

    if (num_lay != 1) and (num_cls > 1):
        print('Error, you have multiple layers and multiple concetrated layers!')
        return;
    else:
        # Setup Graphs
        for i in range(num_cls):
            subbatch = batch[batch['cL'] == cLayers[i]]
            hidden_nodes = subbatch['node_count'].iloc[0]
            loss = subbatch['loss']
            epoch = subbatch['epoch']
            VC = subbatch['VCdim'].iloc[0]
            W = subbatch['param_count'].iloc[0]
            L = subbatch['depth'].iloc[0]
            accuracy = subbatch['binary_accuracy']
            cL = subbatch['cL'].iloc[0]

            node_string = ''
            i = 1
            x = 1
            while i < L:
                node_string = node_string + r'$h_%s$ = ' % i + str(hidden_nodes[i-1]) + ', '
                i += 1

            # Build Sub-Graphs
            key_str = "$VC_{max}$: upper bound of VC dimension \n$W$: total # of network parameters \n $h_i$: number of nodes in hidden layer $i$"
            ax = plt.subplot(2, 1, 1)
            cdict = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys', 'RdPu','Copper']
            colors = np.r_[np.linspace(0.3, .6, 3), np.linspace(0.2, .7, 3)]
            mymap = plt.get_cmap(cdict[count])
            # get the colors from the color map
            my_colors = mymap(colors)
            plt.scatter(epoch,loss, s =.7, color=my_colors[2-(cL-1)], label=(
                    r'$\bf VC_{max}$ = ' + r'%.2E' % Decimal(str(VC)) + r'   $\bf W$ = ' + '%.2E' % Decimal(
                str(W)) + '   ' + node_string + r'   $\bf cL$ = ' + str(cL)))
            # plt.title('Network Depth = '+str(L),fontsize=15,loc='left')
            if legend:
                plt.legend(loc='upper right', markerscale=5, bbox_to_anchor=(1, 1.35),
                           ncol=1, fancybox=True, shadow=True, fontsize=9,
                           title=r'$\bf Network Parameters$' + '\n          (depth=' + str(L) + ')',prop=fontP)

            if x == 1:
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                plt.text(0, 1.25, key_str, fontsize=10,
                         horizontalalignment='left',
                         verticalalignment='top', bbox=props, transform=ax.transAxes)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.grid(color='gray', linestyle='--', linewidth=.5)
            plt.subplot(2, 1, 2)
            plt.scatter(epoch, accuracy, s=.7,color=my_colors[2-(cL-1)])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.grid(color='gray', linestyle='--', linewidth=.5)

    return;
df = pd.DataFrame()
os.chdir("./data")
for file in glob.glob("result_data*"):
    data = pickle.load(open(file, "rb"))
    data = data[data['epoch'].apply(lambda x: x % 10 == 0)]
    df = df.append(data)
depths = [2, 3, 4]
os.chdir("./..")
# df[df['node_count'].apply(lambda x: type(x) != int)]
legend = True
for L in depths:
    plt_df = df[df['node_count'].apply(lambda x: type(x) == int)]
    plt_df = plt_df[plt_df['depth'] == L]
    fig = plt.figure(figsize=[9.5, 8.0], dpi=100)
    # ax = plt.subplot(2, 1, 1)
    # epoch = np.linspace(0,50000,5000)
    # lin_cov = 1/np.exp(epoch)
    # plt.scatter(epoch, lin_cov, s=.1, label=('Linear Convergence'))
    count=0
    for index,group in plt_df.groupby('VCdim'):
        plot_data(group,count,legend)
        count = count +1

    plt.savefig(r'./images/Non_Custom_depth_' + str(L) + '.png', dpi=1000)

for L in depths:
    plt_df = df[df['node_count'].apply(lambda x: type(x) != int)]
    plt_df = plt_df[plt_df['depth'] == L]
    fig = plt.figure(figsize=[9.5, 8.0], dpi=100)
    count = 0
    for index,group in plt_df.groupby('VCdim'):
        customWeightBatch(group,count,legend)
        count = count + 1
    plt.savefig(r'./images/Custom_depth_' + str(L) + '.png', dpi=1000)

legend = False
for L in depths:
    plt_df = df[df['node_count'].apply(lambda x: type(x) == int)]
    plt_df = plt_df[plt_df['depth'] == L]
    fig = plt.figure(figsize=[9.5, 8.0], dpi=100)
    # ax = plt.subplot(2, 1, 1)
    # epoch = np.linspace(0,50000,5000)
    # lin_cov = 1/np.exp(epoch)
    # plt.scatter(epoch, lin_cov, s=.1, label=('Linear Convergence'))
    count = 0
    for index, group in plt_df.groupby('VCdim'):
        plot_data(group, count, legend)
        count = count + 1

    plt.savefig(r'./images/Non_Custom_depth_' + str(L) + 'No_legend.png', dpi=1000)

for L in depths:
    plt_df = df[df['node_count'].apply(lambda x: type(x) != int)]
    plt_df = plt_df[plt_df['depth'] == L]
    fig = plt.figure(figsize=[9.5, 8.0], dpi=100)
    count = 0
    for index, group in plt_df.groupby('VCdim'):
        customWeightBatch(group, count, legend)
        count = count + 1

    plt.savefig(r'./images/Custom_depth_'+str(L)+'No_legend.png', dpi=1000)