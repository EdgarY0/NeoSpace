#ifndef MultiHeadAttention
#define MultiHeadAttention
#include "Funcoes.h"
#include <bits/stdc++.h>
using namespace std;

void applyDownTriangularMatrix(vector<vector<float>>& matrix) {
    for (int i = 0; i < (int)matrix.size(); i++)
        for (int j = i + 1; j < (int)matrix[i].size(); j++)
            matrix[i][j] = -INFINITY;
}

void softmax(vector<vector<float>>& logits) {
    for (auto& row : logits) {
        float max_val = *max_element(row.begin(), row.end());
        float sum_exp = 0.0f;
        for (auto& val : row) {
            val = exp(val - max_val);
            sum_exp += val;
        }
        for (auto& val : row) val /= sum_exp;
    }
}

vector<vector<float>> multiHeadAttention(
    const vector<vector<float>>& X,  // [seq, d_model]
    int heads,
    bool apply_mask,
    const vector<vector<float>>& Wq, // [d_model, heads*head_dim] ou [d_model, d_model]
    const vector<vector<float>>& Wk,
    const vector<vector<float>>& Wv,
    const vector<vector<float>>& Wo // [heads*head_dim, d_model] ou [d_model, d_model]
    ){
    int seq_len = (int)X.size();
    int d_model = (int)X[0].size();
    int qkv_out = (int)Wq[0].size(); // normalmente = d_model
    int head_dim = qkv_out / heads;

    // Q,K,V 
    vector<vector<float>> Q;
    vector<vector<float>> K;
    vector<vector<float>> V;

    multiplicacaoMatrizes(X, Wq, Q);
    multiplicacaoMatrizes(X, Wk, K);
    multiplicacaoMatrizes(X, Wv, V);

    // reshape: [seq, heads, head_dim]
    auto split_heads = [&](const vector<vector<float>>& T){
        vector<vector<vector<float>>> out(heads,
            vector<vector<float>>(seq_len, vector<float>(head_dim)));
            for (int t=0; t<seq_len; ++t){
                for (int h=0; h<heads; ++h){
                    for (int d=0; d<head_dim; ++d){
                        out[h][t][d] = T[t][h*head_dim + d];
                    }
                }
            }
            return out;
        };
        auto Qs = split_heads(Q);
        auto Ks = split_heads(K);
        auto Vs = split_heads(V);
        
        // atenção por head
        vector<vector<vector<float>>> heads_out;
        heads_out.reserve(heads);
        for (int h=0; h<heads; ++h) {
            vector<vector<float>> K_t, scores;
            transpor(Ks[h], K_t); // [head_dim, seq]
            multiplicacaoMatrizes(Qs[h], K_t, scores); // [seq, seq]

        float scale = sqrt((float)head_dim);
        for (auto& row : scores) {
            for (auto& v : row){
                v /= scale;
            }
        }
         
        if (apply_mask) applyDownTriangularMatrix(scores);
        softmax(scores);

        vector<vector<float>> head_out;
        multiplicacaoMatrizes(scores, Vs[h], head_out); // [seq, head_dim]
        heads_out.push_back(move(head_out));
    }

    // concatena heads: [seq, heads*head_dim]
    auto concat = concatenaMatrizes(heads_out);

    // proj final: [seq, d_model]
    vector<vector<float>> out;
    multiplicacaoMatrizes(concat, Wo, out);
    return out;
}

#endif