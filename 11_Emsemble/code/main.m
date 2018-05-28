clc; clear all;
load ada_data.mat;
[e_train,e_test] = adaboost(X_train,y_train,X_test,y_test,300);
hold on;
plot(e_test);
plot(e_train);
legend('testing error', 'training error');
hold off