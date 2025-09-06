#ifndef Funcoes
#define Funcoes
#include <bits/stdc++.h>
using namespace std;

void adicaoMatrizes(vector<vector<float>>& matriz_embeddings, vector<vector<float>> matriz_pe){
    int seq_len = matriz_embeddings.size();
    int dmodel = matriz_embeddings[0].size();
    for(int i=0; i <seq_len; i++){
        for(int j = 0; j < dmodel; j++){
            matriz_embeddings[i][j] += matriz_pe[i][j];
        }
    }
}

void printaMatriz(vector<vector<float>> matriz){
    int d1 = matriz.size();
    int d2 = matriz[0].size();
    for(int i =0; i <d1; i++){
        for(int j = 0; j < d2; j++){
            cout << matriz[i][j] << " ";
        }
        cout << endl;
    }
}

vector<float> geraBias(int dim, bool bias){
    int bias_size = dim;
    vector<float> bias_(bias_size, 0.0f);

    if(bias){
        random_device rd;
        mt19937 gen(rd());
        float k = sqrt(1.0f / dim);
        uniform_real_distribution<float> dist(-k, k);
        for(int i = 0; i < bias_size; i++){
            bias_[i] = dist(gen);
        }
    }
    return bias_;
}

void multiplicacaoMatrizes(const vector<vector<float>>& a,
                                  const vector<vector<float>>& b,
                                  vector<vector<float>>& c) {
    int m = a.size(), n = a[0].size(), p = b[0].size();
    if (n != (int)b.size())
        throw invalid_argument("Shapes não permitem multiplicação.");

    c.assign(m, vector<float>(p, 0.0f));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];
}

vector<vector<float>> concatenaMatrizes(
        const vector<vector<vector<float>>>& matrizes) {
    
    int heads = matrizes.size();
    int rows = matrizes[0].size();
    int cols = matrizes[0][0].size();
    
    vector<vector<float>> output(rows, vector<float>(heads * cols));
    
    for (int r = 0; r < rows; r++) {
        for (int h = 0; h < heads; h++) {
            for (int c = 0; c < cols; c++) {
                output[r][h*cols + c] = matrizes[h][r][c];
            }
        }
    }
    return output;
}

void transpor(const vector<vector<float>>& input,
                     vector<vector<float>>& output) {
    if (input.empty()) return;
    int m = input.size(), n = input[0].size();
    output.assign(n, vector<float>(m));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            output[j][i] = input[i][j];
}

vector<vector<float>> geraMatrizes(int rows, int cols) {
    vector<vector<float>> W(rows, vector<float>(cols));
    random_device rd;
    mt19937 gen(rd());
    float k = sqrt(1.0f / cols);
    uniform_real_distribution<float> dist(-k, k);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            W[i][j] = dist(gen);
        }
    }
    return W;
}


#endif