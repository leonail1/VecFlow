#include <stdio.h>
#include<chrono>

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <set>

#include <random>
#include <cstring>

#include <cstddef>
#include <mutex>
#include <bitset>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <stack>
#include <cassert>
#include <thread>

// fixed
#ifndef RESERVENUM
#define RESERVENUM 128
#endif
#ifndef SS
#define SS 128
#endif
#ifndef BLK_H
#define BLK_H 16
#endif
#ifndef BLK_W
#define BLK_W 16
#endif
#ifndef MAX_P
#define MAX_P 512
#endif
#ifndef SAMPLE
#define SAMPLE 32
#endif
#ifndef INF_DIS
#define INF_DIS 10000000
#endif
#ifndef SELECT_CAND
#define SELECT_CAND 480
#endif

// K
#ifndef K_SIZE
#define K_SIZE 96
#endif
// DIM
#ifndef DIM_SIZE
#define DIM_SIZE 128
#endif
// TOPM
#ifndef TOPM_SIZE
#define TOPM_SIZE 64
#endif
// FINAL_DEGREE 32
#ifndef FINAL_DEGREE_SIZE
#define FINAL_DEGREE_SIZE 64
#endif

#define BUFFER_SIZE_FOR_CACHED_IO (size_t)1024 * (size_t)1048576
#define BUFFER_SIZE_FOR_CACHED_READ (size_t)1024 * (size_t)102400


void load_data(char* filename, float*& data, unsigned& num,unsigned& dim);

struct Neighbor {
    unsigned id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(unsigned id, float distance, bool f) : id{id}, distance{distance}, flag(f) {}
    Neighbor(const Neighbor& nei){
        id = nei.id;
        distance = nei.distance;
        flag = nei.flag;
    }

    inline bool operator<(const Neighbor &other) const {
        return distance < other.distance;
    }
};

static inline int InsertIntoPool (Neighbor *addr, unsigned k, Neighbor nn);

float compare(float* a, float* b, unsigned dim);

void get_neighbors(float *query, unsigned* final_graph_, float* data_, unsigned ep_, unsigned nd_, unsigned L, unsigned DIM, unsigned K,
    boost::dynamic_bitset<> &flags,
    std::vector<Neighbor> &retset,
    std::vector<Neighbor> &fullset);

void cal_ep(unsigned points, float* data, unsigned& ep_, unsigned* graph, unsigned L, unsigned DIM, unsigned K);

void StoreNSG(const char *filename, unsigned* graph, unsigned width, unsigned ep_, unsigned points, unsigned K);

void StoreVamana(const char *filename, unsigned* graph, unsigned ep_, unsigned points, size_t num_frozen_points, unsigned K);

void StoreNSSG(const char *filename, unsigned* graph, unsigned width, unsigned points, unsigned K);

void StoreDPG(const char *filename, unsigned* graph, unsigned N, unsigned degree, unsigned K);

void StoreCAGRA(const char* filename, unsigned* graph, unsigned points_num, unsigned degree, unsigned K);
