#include <bits/stdc++.h>
#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <set>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <limits>
#include <random>

using namespace std;

void carregaTexto(string& texto, const string& path) {
    ifstream arquivo(path);
    if (!arquivo) {
        cerr << "Não foi possível abrir o arquivo." << endl;
        exit(1);
    }
    texto.assign((istreambuf_iterator<char>(arquivo)),
                 istreambuf_iterator<char>());
    arquivo.close();
}

void tokenizeUniqueChars(const string& texto, vector<string>& caracteres) {
    set<string> UniqueChars;

    // Conversor UTF-8 ↔ UTF-32
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    u32string texto32 = conv.from_bytes(texto);

    for (char32_t ch : texto32) {
        UniqueChars.insert(conv.to_bytes(ch)); // guarda como string UTF-8
    }

    for (const string& ch : UniqueChars) {
        caracteres.push_back(ch);
    }
}

void tokenizeChars(const string& texto, vector<string>& caracteres_texto){
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    u32string texto32 = conv.from_bytes(texto);
    for(char32_t ch: texto32){
        caracteres_texto.push_back(conv.to_bytes(ch));
    }
}


void create_decoder_map(vector<string> caracteres, unordered_map<int, string>& vocab_decoder){
    for(int i = 0; i < caracteres.size(); i++){
        vocab_decoder.insert(make_pair(i, caracteres[i]));
    }
}

void create_encoder_map(vector<string> caracteres, unordered_map<string, int>& vocab_encoder){
    for (int i = 0; i < caracteres.size(); i++){
        vocab_encoder.insert(make_pair(caracteres[i], i));
    }
}

void search_idx(unordered_map<string, int>& vocab_encoder, vector<string>& caracteres_texto, vector<int>& idxs){
    for(const string& ch: caracteres_texto){
        for(const auto& par: vocab_encoder){
            if (ch == par.first){
                idxs.push_back(par.second);
            }
        }
    }
}

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

void adicaomatrizes(vector<vector<float>>& matriz_embeddings, vector<vector<float>> matriz_pe){
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

void multiplicacaomatrizes(const vector<vector<float>>& a,
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

void ApplyDownTriangularMatrix(vector<vector<float>>& matrix) {
    for (int i = 0; i < (int)matrix.size(); i++)
        for (int j = i + 1; j < (int)matrix[i].size(); j++)
            matrix[i][j] = -INFINITY;
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

    multiplicacaomatrizes(X, Wq, Q);
    multiplicacaomatrizes(X, Wk, K);
    multiplicacaomatrizes(X, Wv, V);

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
            multiplicacaomatrizes(Qs[h], K_t, scores); // [seq, seq]

        float scale = sqrt((float)head_dim);
        for (auto& row : scores) {
            for (auto& v : row){
                v /= scale;
            }
        }
         
        if (apply_mask) ApplyDownTriangularMatrix(scores);
        softmax(scores);

        vector<vector<float>> head_out;
        multiplicacaomatrizes(scores, Vs[h], head_out); // [seq, head_dim]
        heads_out.push_back(move(head_out));
    }

    // concatena heads: [seq, heads*head_dim]
    auto concat = concatenaMatrizes(heads_out);

    // proj final: [seq, d_model]
    vector<vector<float>> out;
    multiplicacaomatrizes(concat, Wo, out);
    return out;
}

int main() {
    //Carrega Texto de input.txt

    string texto;
    string path = "input.txt";
    carregaTexto(texto, path);
    cout << texto << endl;

    //Cria os nossos tokens em texto de char
    
    vector<string> caracteres;

    vector<string> caracteres_texto;

    tokenizeUniqueChars(texto, caracteres);
    tokenizeChars(texto, caracteres_texto);
    for(const string& ch: caracteres){
        cout << ch << endl;
    }

    //Cria variáveis para as tabelas de codificação e decodificação
    
    unordered_map<string, int> vocab_encoder;
    unordered_map<int, string> vocab_decoder;
    
    //Reserva espaço na memória suficiente
    
    vocab_encoder.reserve(caracteres.size());
    vocab_decoder.reserve(caracteres.size());
    
    //Cria as tabelas
    
    create_decoder_map(caracteres, vocab_decoder);
    create_encoder_map(caracteres, vocab_encoder);

    if (vocab_decoder.size() == vocab_encoder.size()){
        cout << "\n\n---------- CODIFICADOR -----------\n\n" << endl;
        for(auto& par: vocab_encoder){
            if (par.first == " "){
                cout << "Caractere: " << "space" << "          Indice: " << par.second << endl;
            } else if (par.first == "\n"){
               cout << "Caractere: " << "\\n" << "          Indice: " << par.second << endl;
            } else {
                cout << "Caractere: " << par.first << "          Indice: " << par.second << endl;
            }
        }
        cout << "\n\n---------- DECODIFICADOR -----------\n\n" << endl;
        for(auto& par: vocab_decoder){
            if (par.second == " "){
                cout << "Indice: " << par.first << "    Caractere: " << "space" << endl;
            } else if (par.second == "\n"){
                cout << "Indice: " << par.first << "    Caractere: " << "\\n" << endl;
            } else {
                cout << "Indice: " << par.first << "    Caractere: " << par.second << endl;
            }
        }
    } else {
        cout << "Mapas de codificação e decodificação tem tamanhos diferentes!" << endl;
        exit(1);
    }

    vector<int> idxs;
    idxs.reserve(caracteres_texto.size());
    search_idx(vocab_encoder, caracteres_texto, idxs);
    cout << "\n\nLista de indices para matriz de embeddings: ";
    for(auto const& idx: idxs){
        cout << idx << " ";
    }
    cout << "\n\n\n";
    vector<vector<float>> embeddings;
    vector<vector<float>> vocab_embeddings;
    
    int embedding_dim = 16;
    int vocab_size = caracteres.size();
    int seq_len = caracteres_texto.size();
    cout << seq_len << endl;
    int heads = 4;
    
    float k = sqrt(1.0f / embedding_dim);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> distr(-k, k);
    
    for (int i = 0; i < vocab_size; i++) {
        vector<float> embedding_gerado;
        embedding_gerado.reserve(embedding_dim);
        for (int j = 0; j < embedding_dim; j++) {
            embedding_gerado.push_back(distr(gen));
        }
        vocab_embeddings.push_back(move(embedding_gerado));
    }
    
    // Para geração da matriz de embeddings a partir do input
    embeddings.reserve(seq_len);
    
    for(int i = 0; i < seq_len; i++){
        int id = idxs[i];
        if(id < 0 || id >= vocab_size){
            throw out_of_range("Token id fora do vocabulario");
        }
        embeddings.push_back(vocab_embeddings[id]);
    }
    
    cout << "\n\n---------- Matriz Embeddings ----------\n" << endl;
    printaMatriz(embeddings);
    
    
    vector<vector<float>> matriz_pe;
    matriz_pe.reserve(seq_len);

    for(int pos = 0; pos < seq_len; pos++){
        vector<float> linha_pe;
        for (int i = 0; i < embedding_dim; i++){
            float expoente = (float)2*i/(float)embedding_dim;
            float denominador = pow(10000, expoente);
            float valor = (float)pos/denominador;
            if(i % 2 == 0){
                linha_pe.push_back(sinf(valor));
            } else {
                linha_pe.push_back(cosf(valor));
            }
        }
        matriz_pe.push_back(linha_pe);
    }

    for (int i = 0; i <seq_len; i++){
        for (int j =0; j <embedding_dim; j++){
            embeddings[i][j]+= matriz_pe[i][j];
        }
    }


    generate_positional_encoding(matriz_pe, embeddings);
    cout << "\n\n---------- Matriz Positional Encoding ----------\n\n" << endl;
    printaMatriz(matriz_pe);
    adicaomatrizes(embeddings, matriz_pe);
    cout << "\n\n---------- Matriz Embeddings + Positional Encoding ----------\n\n" << endl;
    printaMatriz(embeddings);

    vector<vector<float>> Wq;
    vector<vector<float>> Wk;
    vector<vector<float>> Wv;
    vector<vector<float>> Wo;
    vector<vector<float>> attention_output;

    Wq.reserve(embedding_dim);
    Wk.reserve(embedding_dim);
    Wv.reserve(embedding_dim);
    Wo.reserve(embedding_dim);
    attention_output.reserve(embedding_dim);

    Wq = geraMatrizes(embedding_dim, embedding_dim);
    Wk = geraMatrizes(embedding_dim, embedding_dim);
    Wv = geraMatrizes(embedding_dim, embedding_dim);
    Wo = geraMatrizes(embedding_dim, embedding_dim);

    attention_output = multiHeadAttention(embeddings, heads, true, Wq, Wk, Wv, Wo);

    cout << "\n\n---------- Matriz Attention Output ----------\n\n" << endl;
    printaMatriz(attention_output);

    return 0;
}