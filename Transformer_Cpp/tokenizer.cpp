#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <set>
#include <iostream>
#include <fstream>
#include <unordered_map>

using namespace std;

void carregaTexto(string& texto, const string& path){
    ifstream arquivo(path);
    if(!arquivo){
        cerr << "Não foi possivel abrir o arquivo." << endl;
        exit(1);
    }
    texto.assign((istreambuf_iterator<char>(arquivo)), istreambuf_iterator<char>());
    arquivo.close();
}
//---------------------------------------------------------------------------------
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
//---------------------------------------------------------------------------------
void tokenizeChars(const string& texto, vector<string>& caracteres_texto){
    wstring_convert<codecvt_utf8<char32_t>, char32_t> conv;
    u32string texto32 = conv.from_bytes(texto);
    for(char32_t ch: texto32){
        caracteres_texto.push_back(conv.to_bytes(ch));
    }
}
//---------------------------------------------------------------------------------
void create_decoder_map(vector<string> caracteres, unordered_map<int, string>& vocab_decoder){
    for(int i = 0; i < caracteres.size(); i++){
        vocab_decoder.insert(make_pair(i, caracteres[i]));
    }
}
//---------------------------------------------------------------------------------
void create_encoder_map(vector<string> caracteres, unordered_map<string, int>& vocab_encoder){
    for (int i = 0; i < caracteres.size(); i++){
        vocab_encoder.insert(make_pair(caracteres[i], i));
    }
}
//---------------------------------------------------------------------------------
void search_idx(unordered_map<string, int>& vocab_encoder, vector<string>& caracteres_texto, vector<int>& idxs){
    for(const string& ch: caracteres_texto){
        for(const auto& par: vocab_encoder){
            if (ch == par.first){
                idxs.push_back(par.second);
            }
        }
    }
}
// Main: --------------------------------------------------------------------------
int main(){
    //Carrega Texto de input.txt
    string texto;
    string path = "input.txt";
    carregaTexto(texto,path);
    cout << texto << endl;

    //Cria os nossos tokens em texto de char

    vector<string> caracteres;
    vector<string> caracteres_texto;
    tokenizeUniqueChars(texto, caracteres);
    tokenizeChars(texto, caracteres_texto);
    return 0;   
}