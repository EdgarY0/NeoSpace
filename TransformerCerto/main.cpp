#include "Funcoes.h"
#include "Tokenizer.h"
#include "Embedding.h"
#include "PositionalEnconding.h"
#include "MultiHeadAttention.h"

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
            }
            else if (par.first == "\n"){
               cout << "Caractere: " << "\\n" << "          Indice: " << par.second << endl;
            }
            else {
                cout << "Caractere: " << par.first << "          Indice: " << par.second << endl;
            }
        }
        cout << "\n\n---------- DECODIFICADOR -----------\n\n" << endl;
        for(auto& par: vocab_decoder){
            if (par.second == " "){
                cout << "Indice: " << par.first << "    Caractere: " << "space" << endl;
            } 
            else if (par.second == "\n"){
                cout << "Indice: " << par.first << "    Caractere: " << "\\n" << endl;
            } 
            else {
                cout << "Indice: " << par.first << "    Caractere: " << par.second << endl;
            }
        }
    } 
    else {
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
    adicaoMatrizes(embeddings, matriz_pe);
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

    attention_output = MultiHeadAttention(embeddings, heads, true, Wq, Wk, Wv, Wo);

    cout << "\n\n---------- Matriz Attention Output ----------\n\n" << endl;
    printaMatriz(attention_output);

    return 0;
}