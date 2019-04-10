/* Load the network. */
caffe::NetParameter proto;
caffe::NetParameter net_txt;

ifstream fin;
fin.open(trained_file, std::ios::in | std::ios::binary);
string fileContent;
if(fin.is_open()){
    ostringstream temp;
    temp << fin.rdbuf();
    fileContent = temp.str();
    fin.close();
}
int fileLen = static_cast<int>(fileContent.length());


ifstream tfin;
tfin.open(model_file, std::ios::in | std::ios::binary);
string netContent;
if(tfin.is_open()){
    ostringstream txttemp;
    txttemp << tfin.rdbuf();
    netContent = txttemp.str();
    tfin.close();
}
int netLen = static_cast<int>(netContent.length());


// 解密，首先读取文件尺寸，该参数存放在文件最后10位里
int realFileLen = std::stoi(fileContent.substr(fileLen - 10, 10));
int realNetLen = std::stoi(netContent.substr(netLen - 10, 10));

int realFileLen2 = fileLen - 10;    //16的整数倍
int realNetLen2 = netLen - 10;    //16的整数倍

auto *fileBuf = new unsigned char[realFileLen2];
auto *netBuf = new unsigned char[realNetLen2];
memset(fileBuf, 0, realFileLen2);
memset(netBuf, 0, realNetLen2);

fileContent.copy(reinterpret_cast<char*>(fileBuf), realFileLen2);
netContent.copy(reinterpret_cast<char*>(netBuf), realNetLen2);

unsigned char key[] =
        {
                0x43, 0x65, 0x56, 0x16,
                0x54, 0xae, 0xd2, 0xa5,
                0xab, 0xf7, 0x15, 0x76,
                0x09, 0xef, 0x6f, 0x4c,
        };
AES aes(key);
AES aes_net(key);
aes.InvCipher((void *)fileBuf, realFileLen);
aes_net.InvCipher((void *)netBuf, realNetLen);

// 读取deploy.txt文件内容
string fileBufStr, netBufStr;
fileBufStr.assign(reinterpret_cast<char*>(fileBuf), realFileLen);
netBufStr.assign(reinterpret_cast<char*>(netBuf), realNetLen);


// 读取deploy.prototxt文件内容
//istringstream filenet(netContent);
istringstream filenet(netBufStr);
IstreamInputStream* me_input = new IstreamInputStream((std::istream* )(&filenet));
ZeroCopyInputStream* coded_input = me_input;
google::protobuf::TextFormat::Parse(coded_input, &net_txt);



// 读取xx.caffemodel文件内容
//istringstream netpare(fileContent);
istringstream netpare(fileBufStr);
IstreamInputStream* net_input = new IstreamInputStream((std::istream* )(&netpare));
CodedInputStream* coded_input_p = new CodedInputStream(net_input);
coded_input_p->SetTotalBytesLimit(INT_MAX, 536870912);
proto.ParseFromCodedStream(coded_input_p);

net_.reset(new Net<float>(net_txt));
net_->CopyTrainedLayersFrom(proto);
