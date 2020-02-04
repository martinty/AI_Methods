#pragma once
#include "TextColor.h"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
using namespace Eigen;
using namespace std;

Vector2d forward(Matrix2d om, Matrix2d tm, Vector2d f, int u);
Vector2d backward(Matrix2d om, Matrix2d tm, Vector2d b, int u);
vector<Vector2d> forwardBackward(Matrix2d om, Matrix2d tm, Vector2d prior,
                                 vector<int> ev);

void partB(Matrix2d om, Matrix2d tm, Vector2d rp, vector<int> ev);
void PartC(Matrix2d om, Matrix2d tm, Vector2d rp, vector<int> ev);

Vector2d normalizeSum(Vector2d vec);
Vector2d multiply(Vector2d a, Vector2d b);
void printMsg(string name, int day, Vector2d msg);