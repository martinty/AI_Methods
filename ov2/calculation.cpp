#include "calculation.hpp"

Vector2d forward(Matrix2d om, Matrix2d tm, Vector2d f, int u) {
    if (!u) om = Matrix2d::Identity() - om;
    return normalizeSum(om * tm.transpose() * f);  // Eq. 15.12
}

Vector2d backward(Matrix2d om, Matrix2d tm, Vector2d b, int u) {
    if (!u) om = Matrix2d::Identity() - om;
    return tm * om * b;  // Eq. 15.13
}

vector<Vector2d> forwardBackward(Matrix2d om, Matrix2d tm, Vector2d prior,
                                 vector<int> ev) {
    int size = (int)ev.size();
    vector<Vector2d> fv{prior};
    vector<Vector2d> sv(size);
    Vector2d b;
    b << 1, 1;
    for (int i = 1; i <= size; i++) {
        fv.push_back(forward(om, tm, fv[i - 1], ev[i - 1]));
    }
    for (int i = size; i >= 1; i--) {
        sv[i - 1] = normalizeSum(multiply(fv[i], b));
        b = backward(om, tm, b, ev[i - 1]);
        printMsg("BACKWARD", i, b);
    }
    return sv;
}

void partB(Matrix2d om, Matrix2d tm, Vector2d rp, vector<int> ev) {
    cout << "Part B - Implemented filtering using the FORWARD operation\n\n";
    int day = 0;
    for (auto umbrella : ev) {
        rp = forward(om, tm, rp, umbrella);
        day++;
        printMsg("FORWARD", day, rp);
    }
}

void PartC(Matrix2d om, Matrix2d tm, Vector2d rp, vector<int> ev) {
    cout << "Part C - Implemented smoothing using the FORWARD-BACKWARD "
            "algorithm\n\n";
    vector<Vector2d> smoothing = forwardBackward(om, tm, rp, ev);
    printMsg("FORWARD-BACKWARD", 1, smoothing[0]);
}

Vector2d normalizeSum(Vector2d vec) { return vec / vec.sum(); }

Vector2d multiply(Vector2d a, Vector2d b) {
    for (int i = 0; i < 2; i++) {
        a[i] *= b[i];
    }
    return a;
}

void printMsg(string name, int day, Vector2d msg) {
    cout << TextColor(ColorId::yellow) << setw(24) << name + " message"
         << TextColor() << ":\n"
         << setw(26) << "Day: " << day << "\n"
         << setprecision(3) << fixed << setw(26)
         << "Rain: " << TextColor(ColorId::green) << msg[0] << TextColor()
         << "\n"
         << setw(27) << "¬Rain: " << TextColor(ColorId::red) << msg[1]
         << TextColor() << "\n\n";
}