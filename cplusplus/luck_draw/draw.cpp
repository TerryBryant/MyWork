#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include "draw.h"

using std::cout;
using std::endl;

const int PERSONS_LENGTH = 100;

int lucky_draw(int persons[]) {
    static std::vector<int> all_ids;

    // a STL random numbers generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> id(0, PERSONS_LENGTH - 1);

    while (true) {
        int id_ret = persons[id(gen)];    // id(id_engine) generates a random int,
        // ranges from 0 to 99, which can be used as an index of persons array

        auto iterator = find(all_ids.begin(), all_ids.end(), id_ret);

        // if you can't find the value in vector, you should put it into the vector, say it's been chosen
        if(iterator == all_ids.end()){
            all_ids.push_back(id_ret);
            return id_ret;
        }

        // if you have chosen all ids
        if(all_ids.size() == PERSONS_LENGTH){
            cout << "You have chosen all persons' id !" << endl;
            return -1;
        }
    }
}