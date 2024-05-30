import geatpy as ea
import numpy as np
import os
import re
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import pickle
import evo_search

###################################################################1. preliminaries###########################################################################

# 1.1 search space

node_level_modules_all = ['None','eig','svd','degree','eig,svd','eig,degree','svd,degree','eig,svd,degree']
attn_level_modules_all = ['None','spe','spatial','nhop','spe,spatial','spe,nhop','spatial,nhop','spe,spatial,nhop']
model_scale_all = ('mini','small','middle','large')
topology_all = ('vanilla','Res','JK','GCNII')
gnn_type_all = ('None','GCN','SAGE','GAT','GATv2','GIN')
gnn_insert_pos_all = ('before','alter','parallel')


# 1.2 decode a individual to hyperparameters # individual(node_level_modules,attn_level_modules,model_scale,topology,gnn_type,gnn_insert_pos)

def decode(vector,dn):
    node_level_modules = node_level_modules_all[vector[0]]
    if vector[0] == 0:
        use_super_node = 'False'
    else:
        use_super_node = 'True'
    attn_level_modules = attn_level_modules_all[vector[1]]
    attn_mask_modules = 'None'
    if vector[1] == 3 or vector[1] == 5 or vector[1] == 6 or vector[1] == 7:
        attn_mask_modules = 'nhop' # 'nhop'
    model_scale = model_scale_all[vector[2]]
    topology = topology_all[vector[3]]
    if vector[4] == 0:
        use_gnn_layers = 'False'
        gnn_type = 'GCN'
    else:
        use_gnn_layers = 'True'
        gnn_type = gnn_type_all[vector[4]]
    
    gnn_insert_pos = gnn_insert_pos_all[vector[5]]

    if vector[0] == 0:
        if vector[1] == 0:
            allstr = "--data_name " + dn + " --topology " + topology + " --model_scale " + model_scale + " --use_super_node " + use_super_node + " --attn_mask_modules " + attn_mask_modules + " --use_gnn_layers " + use_gnn_layers + " --gnn_insert_pos " + gnn_insert_pos + " --gnn_type " + gnn_type
        else:
            allstr = "--data_name " + dn + " --topology " + topology + " --model_scale " + model_scale + " --use_super_node " + use_super_node + " --attn_level_modules " + attn_level_modules + " --attn_mask_modules " + attn_mask_modules + " --use_gnn_layers " + use_gnn_layers + " --gnn_insert_pos " + gnn_insert_pos + " --gnn_type " + gnn_type
    else:
        if vector[1] == 0:
            allstr = "--data_name " + dn + " --topology " + topology + " --model_scale " + model_scale + " --use_super_node " + use_super_node + " --node_level_modules " + node_level_modules + " --attn_mask_modules " + attn_mask_modules + " --use_gnn_layers " + use_gnn_layers + " --gnn_insert_pos " + gnn_insert_pos + " --gnn_type " + gnn_type
        else:
            allstr = "--data_name " + dn + " --topology " + topology + " --model_scale " + model_scale + " --use_super_node " + use_super_node + " --node_level_modules " + node_level_modules + " --attn_level_modules " + attn_level_modules + " --attn_mask_modules " + attn_mask_modules + " --use_gnn_layers " + use_gnn_layers + " --gnn_insert_pos " + gnn_insert_pos + " --gnn_type " + gnn_type
    # print(allstr)
    return allstr

# 1.3 encode hyperparameters to a individual
def encode(path):
    # data_name,model_scale,use_super_node,node_level_modules,
    # eig_pos_dim,svd_pos_dim,attn_level_modules,attn_mask_modules,
    # num_hop_bias,use_gnn_layers,gnn_insert_pos,num_gnn_layers,
    # gnn_type,gnn_dropout,ampling_algo,depth,num_neighbors,seed,topology
    surrogate_data_x = []
    surrogate_data_y = []
    for file_name in os.listdir(path):
        # get input
        ind = file_name.split('_')
        if ind[0] in ["COX2", "BZR", "PTC", "DHFR"]:
            ind[4] = ind[4].replace('+',',')
            ind[7] = ind[7].replace('+',',')
            print(ind)
            node_level_modules_ind = node_level_modules_all.index(ind[4])
            attn_level_modules_ind = attn_level_modules_all.index(ind[7])
            model_scale_ind = model_scale_all.index(ind[2])
            topology_ind = topology_all.index(ind[19])
            gnn_type_ind = gnn_type_all.index(ind[13])
            gnn_insert_pos_ind = gnn_insert_pos_all.index(ind[11])
        else:
            ind[3] = ind[3].replace('+',',')
            ind[6] = ind[6].replace('+',',')
            print(ind)
            node_level_modules_ind = node_level_modules_all.index(ind[3])
            attn_level_modules_ind = attn_level_modules_all.index(ind[6])
            model_scale_ind = model_scale_all.index(ind[1])
            topology_ind = topology_all.index(ind[18])
            gnn_type_ind = gnn_type_all.index(ind[12])
            gnn_insert_pos_ind = gnn_insert_pos_all.index(ind[10])
        ind_vector = [node_level_modules_ind,attn_level_modules_ind,model_scale_ind,topology_ind,gnn_type_ind,gnn_insert_pos_ind]
        surrogate_data_x.append(ind_vector)
        # print(ind_vector)

        # get output
        output_all = []
        with open(path+ '/' + file_name + '/test.txt', 'r') as f:
            line = f.readline()
            ent=0       
            while line:       
                print(line)
                if ent < 6:
                    line_s = re.findall(r"\d+\.?\d*",line)
                    if line_s:
                        output_all.append(float(line_s[0]))
                    else:
                        output_all.append(1.0)
                elif ent == 6:
                    line_s = re.findall(r":(.+?)\n",line)
                    output_all.append(line_s[0])
                line = f.readline()
                ent += 1     

        # output_all [predict_loss,predict_MAE,predict_runtime,predict_samples_per_second,predict_samples_per_second,best_val_metric,best_model_checkpoint]
        surrogate_data_y.append(output_all)

        
    return surrogate_data_x,surrogate_data_y

###################################################################2. construct surrogate models###########################################################################

# 2.1 obtain surrogate_data
def sample_data(dataname, NIND, readpath):
    # denote problems [node_level_modules,attn_level_modules,model_scale,topology,gnn_type,gnn_insert_pos]
    problem = ea.Problem(name='GTNAS', 
                        M=1, 
                        maxormins=[1], 
                        Dim=6, 
                        varTypes=[1, 1, 1, 1, 1, 1],
                        lb=[0, 0, 0, 0, 0, 0],
                        ub=[7, 7, 3, 3, 5, 2])
    
    # random a population
    prophetPop = ea.Population(Encoding='RI', Field=(problem.varTypes, problem.ranges, problem.borders), NIND=NIND)
    prophetPop.initChrom(NIND)

    # train data to construct surrogate model
    for ind in prophetPop.Chrom:
        allstr = decode(ind,dataname)
        print(allstr)
        os.system("python run.py " + allstr + " --data_name " + dataname + " --num_train_epochs 10" + " --warmup_steps 4000" + ' --output_dir ' + readpath)

# 2.2 construct surrogate models by different methods
# 1.decision tree regression
model_decision_tree_regression = tree.DecisionTreeRegressor()

# 2.linear regression
model_linear_regression = LinearRegression()

# 3.SVM regression
model_svm = svm.SVR()

# 4.kNN regression
model_k_neighbor = neighbors.KNeighborsRegressor()

# 5.random forest regression
model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=230,max_features='sqrt')

# 6.Adaboost regression
model_adaboost_regressor = ensemble.AdaBoostRegressor()

# 7.GBRT regression
model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor()

# 8.Bagging regression
model_bagging_regressor = ensemble.BaggingRegressor()

# 9.ExtraTree regression
model_extra_tree_regressor = ExtraTreeRegressor()

# 10.Gaussian Process Regression
model_gaussian_process_regressor = GaussianProcessRegressor()

# 11.MLP Regression
model_MLP_regressor = MLPRegressor()

def calculate_MSE(x, y):
    # input two list, x: predict, y: ground truth
    # output MSE
    mse_list = np.array([(element_x - element_y) ** 2 for element_x, element_y in zip(x, y)])
    mse = np.mean(mse_list)
    return mse

def N_K(target, predict, k):
    out_1 = np.argsort(-target)
    out_2 = np.argsort(-predict)
    best_k = out_2[0:k]
    rank = []
    for i in best_k:
        a = np.argwhere(out_1 == i)[0][0] + 1
        rank.append(a)
    return min(rank)

def try_different_method(x_train, y_train, x_test, y_test, model, method, return_flag=False):
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    result = list(result)
    score = r2_score(y_test, result)
    result_arg = np.argsort(result)
    y_test_arg = np.argsort(y_test)
    result_rank = np.zeros(len(y_test_arg))
    y_test_rank = np.zeros(len(y_test_arg))
    for i in range(len(y_test_arg)):
        result_rank[result_arg[i]] = i
        y_test_rank[y_test_arg[i]] = i
    KTau, _ = kendalltau(result_rank, y_test_rank)
    MSEE = calculate_MSE(y_test, result)
    print('method: {:}, KTau: {:}, MSE: {:}, R2score: {:}'.format(method, KTau, MSEE, score))
    print('N@5: {:}, N@10: {:}'.format(N_K(np.array(y_test), np.array(result), 5),
                                       N_K(np.array(y_test), np.array(result), 10)))
    print('--------------------try-end---------------------\n')
    if return_flag:
        return model,KTau, MSEE
    

# 2.3 construct surrogate_data
# data_x [node_level_modules,attn_level_modules,model_scale,topology,gnn_type,gnn_insert_pos]
# data_y [predict_loss,predict_MAE,predict_runtime,predict_samples_per_second,predict_samples_per_second,best_val_metric,best_model_checkpoint]
def con_surr(path_data,path_model,dataname,indd=0):
    # readpath = './surr_outputs'
    surrogate_data = encode(path_data)
    surrogate_data=np.array(surrogate_data)
    np.save(path_model + '/surrogate_data_' + dataname + '_run_' + str(indd) + '.npy', surrogate_data)
    
    # train surrogate model
    x_train = []
    for sdd in surrogate_data[0]:
        x_train.append(np.array(sdd))
    x_train = np.array(x_train)
    best_val_metric = []
    for i in surrogate_data[1]:
        best_val_metric.append(i[5])
    y_train = np.array(best_val_metric)

    # train surrogate_models
    model = [model_decision_tree_regression, model_linear_regression, model_svm, model_k_neighbor,
         model_random_forest_regressor, model_adaboost_regressor, model_gradient_boosting_regressor,
         model_bagging_regressor, model_extra_tree_regressor, model_gaussian_process_regressor, model_MLP_regressor]
    method = ['decision_tree', 'linear_regression', 'svm', 'knn', 'random_forest', 'adaboost', 'GBRT', 'Bagging', 'ExtraTree', 'Gaussian_Process', 'MLP']

    best_KTau = 0
    for i in range(len(model)):
        # train model
        model_save,KTau, MSE = try_different_method(x_train[:18,:], y_train[:18], x_train[18:,:], y_train[18:], model[i], method[i], return_flag=True)
        if KTau > best_KTau:
            best_model_save = model_save
            best_KTau = KTau
    # save model
    with open(path_model + '/train_model_' + dataname + '_run_' + str(indd) + '.pkl', 'wb') as f:
        pickle.dump(best_model_save, f)

# Ablation studies
def con_surr_all(path_data,path_model,dataname,indd=0):
    # readpath = './surr_outputs'
    surrogate_data = encode(path_data)
    surrogate_data=np.array(surrogate_data)
    np.save(path_model + '/surrogate_data_' + dataname + '_run_' + str(indd) + '.npy', surrogate_data)
    
    # train surrogate model
    x_train = []
    for sdd in surrogate_data[0]:
        x_train.append(np.array(sdd))
    x_train = np.array(x_train)
    best_val_metric = []
    for i in surrogate_data[1]:
        best_val_metric.append(i[5])
    y_train = np.array(best_val_metric)

    # train surrogate_models
    model = [model_decision_tree_regression, model_linear_regression, model_svm, model_k_neighbor,
         model_random_forest_regressor, model_adaboost_regressor, model_gradient_boosting_regressor,
         model_bagging_regressor, model_extra_tree_regressor, model_gaussian_process_regressor, model_MLP_regressor]
    method = ['decision_tree', 'linear_regression', 'svm', 'knn', 'random_forest', 'adaboost', 'GBRT', 'Bagging', 'ExtraTree', 'Gaussian_Process', 'MLP']

    # best_KTau = 0
    KTau_all = []
    MSE_all = []
    
    test_index = np.random.choice(np.arange(len(y_train)),size=round(len(y_train)*0.1),replace=False)
    train_index = np.delete(np.arange(len(y_train)),test_index)

    for i in range(len(model)):
        # train model
        model_save,KTau,MSE = try_different_method(x_train[train_index,:], y_train[train_index], x_train[test_index,:], y_train[test_index], model[i], method[i], return_flag=True)
        KTau_all.append(KTau)
        MSE_all.append(MSE)
    # # save model
    # with open(path_model + '/train_model_' + dataname + '_run_' + str(indd) + '.pkl', 'wb') as f:
    #     pickle.dump(best_model_save, f)
    return KTau_all, MSE_all

##################################################################3. search process#######################################################################

# 3.1 evaluate all individuals in the population by Surrogate model
# denote problems [node_level_modules,attn_level_modules,model_scale,topology,gnn_type,gnn_insert_pos]
class Myproblem(ea.Problem):
    def __init__(self, model, maxormins):
        name = 'GTNAS'  
        M = 1  
        maxormins = maxormins  
        Dim = 6 
        varTypes = [1, 1, 1, 1, 1, 1] 
        lb = [0, 0, 0, 0, 0, 0] 
        ub = [7, 7, 3, 3, 5, 2]
        lbin = [1] * Dim 
        ubin = [1] * Dim 

        ea.Problem.__init__(self,
                            name,
                            M,
                            maxormins,
                            Dim,
                            varTypes,
                            lb,
                            ub,
                            lbin,
                            ubin)
        self.model = model

    def evalVars(self, Vars):
        ObjV = self.model.predict(Vars)
        ObjV = np.reshape(ObjV,(-1,1))
        return ObjV

def search(path_model,dataname,maxormins,indd):
    # load model
    with open(path_model + '/train_model_' + dataname + '_run_' + str(indd) + '.pkl', 'rb') as f:
        model = pickle.load(f)

    problem = Myproblem(model,maxormins)
    algorithm = evo_search.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='RI', NIND=20),
        MAXGEN=20,  
        logTras=1,  
        trappedValue=1e-6, 
        maxTrappedCount=10) 
    
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=0,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=True,
                      dirName='./outputs/'+dataname+'/run_'+str(indd))
    print(res)
    decode(res['Vars'][0],dataname)

def finetune(dataname,indd):
    filename='./outputs/'+ dataname + '/run_' + str(indd) + '/optPop/Chrom.csv'
    with open(filename,'rt') as csvfile:
        data = np.loadtxt(csvfile, delimiter=',').astype(np.int64)
        print(data)
    
    allstr = decode(data,dataname)
    os.system("python run.py " + allstr + " --data_name " + dataname + " --dataloader_num_workers 2" + " --num_train_epochs 100" + " --warmup_steps 40000" + ' --output_dir ./outputs/' + dataname + '/run_' + str(indd))

# 2.4 count surrogate_data
# data_x [node_level_modules,attn_level_modules,model_scale,topology,gnn_type,gnn_insert_pos]
# data_y [predict_loss,predict_MAE,predict_runtime,predict_samples_per_second,predict_samples_per_second,best_val_metric,best_model_checkpoint]
def count_surr(path_data):
    # readpath = './surr_outputs'
    surrogate_data = encode(path_data)
    surrogate_data=np.array(surrogate_data)
    # np.save(path_model + '/surrogate_data_' + dataname + '_run_' + str(indd) + '.npy', surrogate_data)
    
    # train surrogate model
    x_train = []
    for sdd in surrogate_data[0]:
        x_train.append(np.array(sdd))
    # train_x = torch.tensor(sd)[0:15,:]
    x_train = np.array(x_train)
    best_val_metric = []
    for i in surrogate_data[1]:
        best_val_metric.append(i[5])
    y_train = np.array(best_val_metric)
    # train_y = torch.tensor(best_val_metric)[0:15]

    sortnp = np.argsort(y_train)[-2:]
    return x_train[sortnp]

if __name__ == '__main__':

    pattern = 'con_data' # 1. 'con_data': generate the surrogate data, 2. 'search': evolutionary search with surrogate model, 3. 'finetune': retrain for the best arches searched by EGTAS， 4. 'con_surr': construct surrogate model for ablation studies
    dataname = 'ogbg-molhiv' 
    path_data = './surr_data/' + dataname # save surrogate_data，
    if not os.path.exists(path_data):  
        os.makedirs(path_data)
    path_model = './surr_outputs/' # save surrogate model
    if not os.path.exists(path_model):  
        os.makedirs(path_model)

    if pattern == 'con_data':
        # 1 generate the surrogate_data
        data_num = 30 # number of data
        sample_data(dataname, data_num, path_data)
    elif pattern == 'con_surr':
        # 2 construct a surrogate model
        # con_surr(path_data,path_model,dataname)

        KTau_all = []
        MSE_all = []
        for r in range(10):
            KTau, MSE = con_surr_all(path_data,path_model,dataname)
            KTau_all.append(KTau)
            MSE_all.append(MSE)
        KTau_all = np.array(KTau_all)
        MSE_all = np.array(MSE_all)
        method = ['decision_tree', 'linear_regression', 'svm', 'knn', 'random_forest', 'adaboost', 'GBRT', 'Bagging', 'ExtraTree', 'Gaussian_Process', 'MLP']
        print(method)
        print('KTau:')
        KTau_mean = np.mean(KTau_all,axis=0)
        KTau_std = np.std(KTau_all,axis=0)
        min_k = 0
        for i in [0,4,5,7,8,9]:
        # for i in range(len(KTau_mean)):
            # print(KTau_mean[i])
            # print(KTau_std[i])
            print('%.2e(%.2e) &'%(KTau_mean[i],KTau_std[i]))
            if KTau_mean[i] > min_k:
                min_k = KTau_mean[i]
        print(min_k)
        print('MSE:')
        MSE_mean = np.mean(MSE_all,axis=0)
        MSE_std = np.std(MSE_all,axis=0)
        min_m = 1000
        for i in [0,4,5,7,8,9]:
            print("%.2e(%.2e) &"%(MSE_mean[i],MSE_std[i]))
            if MSE_mean[i] < min_m:
                min_m = MSE_mean[i]
        print(min_m)

    elif pattern == 'search':
        # 3 search
        for i in range(10):
            con_surr(path_data,path_model,dataname,i)
            if dataname == 'ZINC':
                maxormins = [1] # maxormins（1：mins；-1：max）
            else:
                maxormins = [-1] # maxormins（1：mins；-1：max）
            search(path_model,dataname,maxormins,i)
            
    elif pattern == 'finetune':
        for i in range(10):
            finetune(dataname,i)
    else:
        print('Input Error! Plase Input: con_data, con_surr, search, finetune.')
