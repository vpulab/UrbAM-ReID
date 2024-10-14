#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np
import math

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]
def dist_vicenty(latitud1,longitud1,latitud2,longitud2):
    '''log1 = (abs(longitud1) * math.pi) / 180
    log2 = (abs(longitud2) * math.pi) / 180
    lat1 = (abs(latitud1) * math.pi) / 180
    lat2 = (abs(latitud2 * math.pi)) / 180
    delta = abs(log1 - log2)
    promedio = (abs(latitud1 + latitud2)) / 2
    R = 1000 * 6379.57 + abs((15 - promedio)) * (0.782);#Arreglo del radio terrestre.
    D = R * math.atan(math.sqrt(math.pow(math.cos(lat2) * math.sin(delta),2) + math.pow(math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta),2)) / ((math.sin(lat1) * math.sin(lat2)) + (math.cos(lat1) * math.cos(lat2) * math.cos(delta))))
'''

    log1 = (np.abs(longitud1) * math.pi) / 180
    log2 = (np.abs(longitud2) * math.pi) / 180
    lat1 = (np.abs(latitud1) * math.pi) / 180
    lat2 = (np.abs(latitud2 * math.pi)) / 180
    delta = np.abs(np.subtract(log1,log2))
    promedio = (np.abs(np.add(latitud1, latitud2))) / 2
    R = 1000 * 6379.57 + np.abs((15 - promedio)) * (0.782)  # Arreglo del radio terrestre.
    D = R * np.arctan(np.sqrt(np.power((np.cos(lat2) * np.sin(delta)), 2) + np.power((
        np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta)), 2)) / (
                                  (np.sin(lat1) * np.sin(lat2)) + (
                                      np.cos(lat1) * np.cos(lat2) * np.cos(delta))))

    return D
def re_rankingGPS(qf, gf, qf_gps, gf_gps,query_rename,gallery_rename, k1=20, k2=6, lambda_value=0.3):

    query_num = len(qf)
    gallery_num = len(gf)

    Misma_GPS={}
    Misma_GPS2={}
    Misma_GPS3={}
    Misma_GPS_id={}
    Misma_GPS_id2={}
    Misma_GPS_id3 = {}
    cont=0
    for i in range(query_num):
        Misma_GPS[qf_gps[i]]=[]
        Misma_GPS_id[i]=[]
        for j in range(query_num):
            if dist_vicenty(qf_gps[i][0], qf_gps[i][1], qf_gps[j][0], qf_gps[j][1]) < 90:
                Misma_GPS[qf_gps[i]].append(qf_gps[j])
                Misma_GPS_id[i].append(j)
                cont=cont+1
        
    cont2 = 0
    for i in range(gallery_num):
        Misma_GPS2[gf_gps[i]]=[]
        Misma_GPS_id2[i]=[]
        for j in range(gallery_num):
            if dist_vicenty(gf_gps[i][0], gf_gps[i][1], gf_gps[j][0], gf_gps[j][1]) < 90:
                Misma_GPS2[gf_gps[i]].append(gf_gps[j])
                Misma_GPS_id2[i].append(j)
                cont2=cont2+1
        
    cont3 = 0
    for i in range(query_num):
        Misma_GPS3[qf_gps[i]]=[]
        Misma_GPS_id3[i]=[]
        for j in range(gallery_num):
            if dist_vicenty(qf_gps[i][0],qf_gps[i][1],gf_gps[j][0],gf_gps[j][1])<90:
                Misma_GPS3[qf_gps[i]].append(gf_gps[j])
                Misma_GPS_id3[i].append(j)

                cont3 = cont3 + 1


    # Inicializa una matriz para almacenar las distancias reorganizadas
    q_g_dist = (np.zeros((query_num, gallery_num), dtype=np.float32))
    q_q_dist = (np.zeros((query_num, query_num), dtype=np.float32))
    g_g_dist = (np.zeros((gallery_num, gallery_num), dtype=np.float32))


    for i in range(query_num):
        for j in Misma_GPS_id[i]:
            q_q_dist[i, j] = np.dot(qf[i], qf[j])
    for i in range(gallery_num):
        for j in Misma_GPS_id2[i]:
            g_g_dist[i, j] = np.dot(gf[i], gf[j])
    for i in range(query_num):
        for j in Misma_GPS_id3[i]:
            q_g_dist[i, j] = np.dot(qf[i], gf[j])
    # for i in range(query_num):
    #     # Gallery que comparten coordenadas GPS con la query actual
    #     relevant_gallery_indices = []  # Almacena los índices de galería relevantes para esta consulta
    #     current_qf_gps = qf_gps[i]  # Coordenadasz GPS de la consulta actual
    #     for j in range(gallery_num):
    #         if np.abs(gf_gps[j][0] - current_qf_gps[0]) <= 0.00019 and np.abs(gf_gps[j][1] - current_qf_gps[1]) <= 0.00019:
    # 
    #             relevant_gallery_indices.append(j)
    # 
    #     # Calcula las distancias solo entre la consulta actual y las galerías relevantes
    #     if relevant_gallery_indices:
    #         current_qf = qf[i]
    #         for j in relevant_gallery_indices:
    #             current_gf = gf[j]
    # 
    #             # Calcula la distancia entre la consulta actual y la galería actual
    #             q_g_dist[i,j] = np.dot(current_qf, current_gf)
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   # change the cosine similarity metric to euclidean similarity metric
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
    
