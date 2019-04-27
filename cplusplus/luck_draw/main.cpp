// 一个抽奖程序，每次从100个id里面随机抽出一个，要求不能重复，而且抽完了也就不抽了
#include <iostream>
#include "draw.h"

using std::cout;
using std::endl;


void Test() {
    int persons[100] = {0};
    for (int i = 0; i < 100; ++i) {
        persons[i] = i;
    }


    for (int i = 0; i < 10; ++i) {
        cout<<lucky_draw(persons)<<endl;
    }
}

int main() {
    Test();

    return 0;
}
