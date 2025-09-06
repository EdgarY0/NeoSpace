#ifndef tokenizer
#define tokenizer
#include <bits/stdc++.h>
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




#endif