//
// Created by lyl on 18-11-27.
//

#ifndef AES_AES_H
#define AES_AES_H

#include "string.h"
class  AES
{
public:
    AES(unsigned char* key);//初始化时输入key
    virtual ~AES();
    unsigned char* Cipher(unsigned char* input);
    unsigned char* InvCipher(unsigned char* input);

    void* Cipher(void* input, int length);//加密
    void* InvCipher(void* input, int length);//解密

private:
    unsigned char Sbox[256];
    unsigned char InvSbox[256];
    unsigned char w[11][4][4];

    void KeyExpansion(unsigned char* key, unsigned char w[][4][4]);
    unsigned char FFmul(unsigned char a, unsigned char b);

    void SubBytes(unsigned char state[][4]);
    void ShiftRows(unsigned char state[][4]);
    void MixColumns(unsigned char state[][4]);
    void AddRoundKey(unsigned char state[][4], unsigned char k[][4]);

    void InvSubBytes(unsigned char state[][4]);
    void InvShiftRows(unsigned char state[][4]);
    void InvMixColumns(unsigned char state[][4]);
};



#endif //AES_AES_H
