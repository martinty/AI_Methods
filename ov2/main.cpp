#include "calculation.hpp"

int main() {
#ifdef __linux__
#else
    system("chcp 65001 > nul");
#endif
    Matrix2d observationModel, transitionModel;
    Vector2d rainProb;
    observationModel << 0.9, 0.0, 0.0, 0.2;
    transitionModel << 0.7, 0.3, 0.3, 0.7;
    rainProb << 0.5, 0.5;
    vector<int> evidence2{1, 1};
    vector<int> evidence5{1, 1, 0, 1, 1};

    partB(observationModel, transitionModel, rainProb, evidence2);
    partB(observationModel, transitionModel, rainProb, evidence5);
    PartC(observationModel, transitionModel, rainProb, evidence2);
    PartC(observationModel, transitionModel, rainProb, evidence5);

    cin.get();
    return 0;
}