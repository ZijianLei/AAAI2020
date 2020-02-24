%module python_extend
%{
extern int train_llsvm(int argc, char **argv);
extern float predict_llsvm(int argc, char **argv);

%}

extern int train_llsvm(int argc, char **argv);
extern float predict_llsvm(int argc, char **argv);