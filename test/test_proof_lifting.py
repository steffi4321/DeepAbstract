import numpy as np
from src.proof_lifting import perform_evaluation_complete, verify_original, get_dirPrefix, cluster_net
from src.main import set_file_model

def compare(network, clusters, epsilon):
    verified, veriIm, cl_time, veri_time, totalIm = perform_evaluation_complete(clusters, epsilon, numImages=50)
    print("results: verifierd", verified, "veriIm", veriIm, "cl_time", cl_time, "veri_time ", veri_time, "totalIm", totalIm)
    dicti = verify_original(network, epsilon, num_tests=50)
    print("results orig: ", dicti)
    return dicti, verified, veriIm, cl_time, veri_time

def mstr(liste):
    ret = '['
    for i in liste:
        ret += str(i) + ';'
    ret = ret[:-1]
    ret += ']'
    return ret


if __name__=='__main__':
    with open('tmp/results_proof_lifting.csv', 'w+') as f:
        f.write('Network,O-VeriTime,O-Prec,O-Veri,O-Pred,A-VeriTime,A-VeriPrec,A-Time,clusters\n')
    cls = [[0,0,0,0,0,20],[0,0,0,0,20,20]]
    network = 'models/MNIST_6x50.tf'
    for clusters in cls:
        #TODO: compare(... layers) layers should be dynamic
        dicti, verified, veriIm, cl_time, veri_time = compare(network, clusters, 0.001)
        with open('tmp/results_proof_lifting.csv','a+') as f:
            line = network[6:-3] + ',' + str(dicti['time']) + ',' + str(dicti['verified']/dicti['total']) + ','
            line += str(dicti['verified']) + ',' + str(dicti['total']) + ','
            line += str(veri_time) + ',' + str(verified/dicti['total']) + ',' + str(cl_time) + ',' + mstr(clusters)
            line += ',' + dicti['verified_list'] + '\n'
            f.write(line)
