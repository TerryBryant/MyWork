#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include "AES.h"


using std::cout;
using std::endl;
using std::string;


int encrypt(const string& src_file, const string& dst_file){
    unsigned char key[] =
            {
                    0x43, 0x65, 0x56, 0x16,
                    0x54, 0xae, 0xd2, 0xa5,
                    0xab, 0xf7, 0x15, 0x76,
                    0x09, 0xef, 0x6f, 0x4c,
            };

    AES aes(key);

    std::string file_content;   // 源文件内容，写入该string中

    std::ifstream fin;
    fin.open(src_file);
    if(fin.is_open()){
        std::ostringstream tmp;
        tmp << fin.rdbuf();
        file_content = tmp.str();
        fin.close();
    } else {
        cout << "Can't open src file: " << src_file << endl;
        return -1;
    }

    // 将string写入unsigned char数组中，用于加密
    int file_len = static_cast<int>(file_content.length());
    int file_len2 = int(ceil(file_len / 16.0) * 16);//注意补零
    auto *file_buf = new unsigned char[file_len2];
    memset(file_buf, 0, file_len2);

    file_content.copy(reinterpret_cast<char*>(file_buf), file_len);
    aes.Cipher((void *)file_buf, file_len);    // 加密

    // 加密后的内容写入新文件
    std::ofstream fout;
    fout.open(dst_file, std::ios::out | std::ios::binary);
    if(fout.is_open()){
        fout.write(reinterpret_cast<char*>(file_buf), file_len2);   // 因为加密的过程中自动补零了，所以加密文件的长度为file_len2
        char dataLen[10] = {0};
        sprintf(dataLen, "%10d", file_len);
        fout.write(dataLen, 10);
        //fout.write(std::to_string(file_len).c_str(), 10);   // 文件的字节数最多10位数，足以存放10G以内
        fout.close();
    } else {
        cout << "Can't open dst file: " << dst_file << endl;
        return -1;
    }

	delete[] file_buf;
    return 0;
}

int main(int argc, char* argv[])
{
//    // aes加密例子
//    // 待加密的内容
//    unsigned char input[] =
//            {
//                    0x32, 0x43, 0xf6, 0xa8,
//                    0x88, 0x5a, 0x30, 0x8d,
//                    0x31, 0x31, 0x98, 0xa2,
//                    0xe0, 0x37, 0x07, 0x34
//            };
//
//    // 自定义密钥
//    unsigned char key[] =
//            {
//                    0x43, 0x65, 0x56, 0x16,
//                    0x54, 0xae, 0xd2, 0xa5,
//                    0xab, 0xf7, 0x15, 0x76,
//                    0x09, 0xef, 0x6f, 0x4c,
//            };
//
//
//    AES aes(key);

//    cout << "加密前的内容是： ";
//    for(int i=0; i<16; i++)
//        cout << (unsigned char)input[i];
//    cout<<endl;
//
//
//    aes.Cipher((void *)input, sizeof(input) / sizeof(unsigned char));    // 加密
//    cout << "加密后的内容是： ";
//    for(int i=0; i<16; i++)
//        cout << (unsigned char)input[i];
//    cout<<endl;
//
//
//
//    aes.InvCipher((void *)input, sizeof(input) / sizeof(unsigned char));    // 解密
//    cout << "解密后的内容是： ";
//    for(int i=0; i<16; i++)
//        cout << (unsigned char)input[i];
//    cout<<endl;

    std::string file_txt = "deploy.prototxt";
    std::string file_model = "deploy.caffemodel";


    std::string file_txt_out = "deploy_aes.prototxt";
    std::string file_model_out = "deploy_aes.caffemodel";


    if(0 != encrypt(file_txt, file_txt_out))
        return -1;

    if(0 != encrypt(file_model, file_model_out))
        return -1;

    return 1;
    
}

