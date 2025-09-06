#ifndef PositionalEncondig
#define PositionalEncondig
#include <bits/stdc++.h>
using namespace std;

void generate_positional_encoding(vector<vector<float>>& matriz_pe, vector<vector<float>> matriz_embeddings){
    int dmodel = matriz_embeddings[0].size();
    int seq_len = matriz_embeddings.size();
    matriz_pe.reserve(seq_len);
    for (int pos =0; pos < seq_len; pos++){
        vector<float> linha_pe;
        for (int i=0; i < dmodel; i++){
            float expoente = (float)2*i/(float)dmodel;
            float potencia = pow(10000, expoente);
            float valor = pos/potencia;
            if (i % 2 == 0){
                linha_pe.push_back(sinf(valor));
            } else {
                linha_pe.push_back(cosf(valor));

            }
        }
        matriz_pe.push_back(linha_pe);
    }
}
#endif