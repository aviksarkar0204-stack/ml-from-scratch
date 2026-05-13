cmake_minimum_required(VERSION 4.1)
project(ML_from_Scratch C)

set(CMAKE_C_STANDARD 11)

add_executable(ML_from_Scratch
        "Linear_regression/Linear_ regression.c"
        Logistic_regression/Logistic_regression.c
        KNN/KNN.cpp
        Polynomial_regression/Polynomial_regression.c
        math_utils.h
        math_utils.c
        main.c
        "Ridge Regression (L2 regularization)/Ridge_regression.c"
        "Lasso Regression (L1 regularization)/Lasso_regression.c")


