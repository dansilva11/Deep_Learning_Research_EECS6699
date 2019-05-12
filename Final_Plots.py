# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import glob
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
                

def equalWeightBatch(batch):
    #Strip out any runs that were not using custom_weights
    batch = batch[batch['custom_weights'].apply(lambda x: not(x))]
    
    layers = batch['depth'].unique()
    num_lay = len(layers)
    params = batch['param_count'].unique()
    num_params = len(params)
    
    for i in range(num_lay):
        for j in range(num_params):
            print('i: ' + str(i) + ' | j: '+ str(j))
            subbatch = batch[batch['depth'] == layers[i]]
            subbatch = subbatch[subbatch['param_count'] == params[j]]
            if(0<len(subbatch)):
                hidden_nodes = subbatch['node_count'].iloc[0]
                subbatch = subbatch.groupby(['epoch']).mean()
                loss = subbatch['loss']
                VC = subbatch['VCdim'].iloc[0]
                W = subbatch['param_count'].iloc[0]
                L = subbatch['depth'].iloc[0]
                accuracy = subbatch['binary_accuracy']
                cL = subbatch['cL'].iloc[0]
                
                # Setup Graphs
                node_string = ''
                k = 1
                x=1
                while k < L:
                    node_string = node_string + r'$h_%s$ = '%k + str(hidden_nodes) +', '
                    k+=1
            
                # Build Sub-Graphs
                key_str = "$VC_{max}$: upper bound of VC dimension \n$W$: total # of network parameters \n $h_i$: number of nodes in hidden layer $i$"
                ax = plt.subplot(2, 1, 1)
                plt.plot(loss, label=(
                            r'$\bf VC_{max}$ = ' + r'%.2E' % Decimal(str(VC)) + r'   $\bf W$ = ' + '%.2E' % Decimal(
                        str(W)) + '   ' + node_string ))
                # plt.title('Network Depth = '+str(L),fontsize=15,loc='left')
            
                plt.legend(loc='upper right', markerscale=50, bbox_to_anchor=(1, 1.35),
                           ncol=1, fancybox=True, shadow=True, fontsize=10,
                           title=r'$\bf Network Parameters$' + '\n          (depth=' + str(L) + ')',prop=   fontP)
            
            
                if x == 1:
                    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                    plt.text(0, 1.25, key_str, fontsize=10,
                             horizontalalignment='left',
                             verticalalignment='top', bbox=props, transform=ax.transAxes)
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.grid(color='gray', linestyle='--', linewidth=.5)
                plt.subplot(2, 1, 2)
                plt.plot(accuracy)
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.grid(color='gray', linestyle='--', linewidth=.5)
            
    return ;
    
def customWeightBatch(batch):
    #Strip out any runs that were not using custom_weights
    batch = batch[batch['custom_weights']]
    
    layers = batch['depth'].unique()
    num_lay = len(layers)
    cLayers = batch['cL'].unique()
    num_cls = len(cLayers)
    #params = batch['param_count'].unique()
    #num_params = len(params)
    
    if (num_lay != 1) and (num_cls > 1):
        print('Error, you have multiple layers '+str(layers)+' and multiple concetrated layers!')
        return ;
    else:
        # Setup Graphs
        for i in range(num_cls):
            subbatch = batch[batch['cL'] == cLayers[i]]
            hidden_nodes = subbatch['node_count'].iloc[0]
            subbatch = subbatch.groupby(['epoch']).mean()
            loss = subbatch['loss']
            VC = subbatch['VCdim'].iloc[0]
            W = subbatch['param_count'].iloc[0]
            L = subbatch['depth'].iloc[0]
            accuracy = subbatch['binary_accuracy']
            cL = subbatch['cL'].iloc[0]
            
            node_string = ''
            i = 0
            x = 1
            while i < L:
                node_string = node_string + r'$h_%s$ = ' % i + str(hidden_nodes[i]) + ', '
                i += 1
        
            # Build Sub-Graphs
            key_str = "$VC_{max}$: upper bound of VC dimension \n$W$: total # of network parameters \n $h_i$: number of nodes in hidden layer $i$"
            ax = plt.subplot(2, 1, 1)
            plt.plot(loss, label=(
                        r'$\bf VC_{max}$ = ' + r'%.2E' % Decimal(str(VC)) + r'   $\bf W$ = ' + '%.2E' % Decimal(
                    str(W)) + '   ' + node_string + r'   $\bf cL$ = ' + str(cL)))
            # plt.title('Network Depth = '+str(L),fontsize=15,loc='left')
        
            plt.legend(loc='upper right', markerscale=50, bbox_to_anchor=(1, 1.35),
                       ncol=1, fancybox=True, shadow=True, fontsize=10,
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
            plt.plot(accuracy)
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.grid(color='gray', linestyle='--', linewidth=.5)

    return ;
    
def plotBatches(batch,title=None):
    import time
    fig = plt.figure(figsize=[9.5, 8.0],dpi=100)
    
    #check if the network has a multi-element array of hidden nodes
    #   if it does, then we know the batch was from Def_Model.buildCustModel() 
    print(batch['node_count'])
    if(isinstance(batch['node_count'].iloc[0],np.ndarray)):
        print('in 1')
        customWeightBatch(batch)
    else:
        print('in 2')
        equalWeightBatch(batch)
    
    plt.savefig('./data/'+title,dpi=1000)
    #plt.show()
    
    return ;

df = pd.DataFrame()
os.chdir("./data")
for file in glob.glob("result_data*"):
   print(file)
   try:
       data = pd.read_pickle(open(file, "rb"))
       data = data[data['epoch'].apply(lambda x: x % 100 == 0)]
       df = df.append(data)
   except Exception as e:
       print(e)
os.chdir("./..")

#append info to data
df['cL'] = df['node_count'].apply(lambda x: np.where(x == np.max(x))[0][0]+1)
df['custom_weights'] = df['node_count'].apply(lambda x: not isinstance(x,int))
     
#get all 2 layer non-custom networks:
mybatch = df[(df['custom_weights'].apply(lambda x:not(x)))]
mybatch =  mybatch[mybatch['depth']==2]
plotBatches(mybatch,'phase_1.png')

#get all equal weight graphs with depth
mybatch = df[(df['custom_weights'].apply(lambda x:not(x)))]
mybatch =  mybatch[mybatch['depth']>2]
plotBatches(mybatch,'phase_2.png')

#get each custom graph per depth
depths = df['depth'].unique()
for d in depths:
    if(d>2):
        mybatch =  df[df['depth']==d]
        mybatch = mybatch[mybatch['custom_weights']]    
        plotBatches(mybatch,'phase_3_depth_'+str(d)+'.png')    
