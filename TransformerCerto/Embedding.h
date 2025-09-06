#ifndef Embedding 
#define Embedding
#include <bits/stdc++.h>
using namespace std;

void search_idx(unordered_map<string, int>& vocab_encoder, vector<string>& caracteres_texto, vector<int>& idxs){
    for(const string& ch: caracteres_texto){
        for(const auto& par: vocab_encoder){
            if (ch == par.first){
                idxs.push_back(par.second);
            }
        }
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

#endif