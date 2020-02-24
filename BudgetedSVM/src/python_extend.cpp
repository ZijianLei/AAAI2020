/*
	\file budgetedsvm-train.cpp
	\brief Source file implementing commmand-prompt interface for training phase of budgetedSVM library.
*/
/*
	Copyright (c) 2013-2014 Nemanja Djuric, Liang Lan, Slobodan Vucetic, and Zhuang Wang
	All rights reserved.

	Author	:	Nemanja Djuric
	Name	:	budgetedsvm_train.cpp
	Date	:	November 30th, 2012
	Desc.	:	Source file implementing commmand-prompt interface for training phase of budgetedSVM library.
*/

#include "../Eigen/Dense"
using namespace Eigen;

#include <vector>
#include <sstream>
#include <time.h>
#include <algorithm>
#include <stdio.h>
using namespace std;

#include "budgetedSVM.h"
#include "llsvm.h"
extern "C"{
int train_llsvm(int argc, char **argv)
//argc 用于统计命令行参数的个数， argv为数组指针，用来存放指向字符串参数的指针
{
    printf("%c",argv);
	parameters param;
	// if argc == 1 print the help file of the program
	if (argc == 1)
	{
		printUsagePrompt(true, &param); //print the usage information trun ->train false -> predict
		return 0;
	}

	// vars
	char inputFileName[1024];
	char modelFileName[1024];
	budgetedModel *model = NULL;

	// parse input string
	parseInputPrompt(argc, argv, true, inputFileName, modelFileName, NULL, &param);
	// modified the parameter for future training
	// first read the parameter and then read the input and output file if the i > argc, which means thbuere are other
	// parameters have not been read

	// init random number generator, if randomization is switched off seed the RNG with a constant
	if (param.RANDOMIZE)
		srand((unsigned)time(NULL));
	else
		srand(0);

	// train a model
	budgetedData *trainData = NULL;

    model = new budgetedModelLLSVM;
    trainData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE);
    trainLLSVM(trainData, &param, (budgetedModelLLSVM*) model);



	// save model to .txt file
	model->saveToTextFile(modelFileName, &(trainData->yLabels), &param);
	delete model;

	// delete trainData, no more need for it
	delete trainData;
}

float predict_llsvm(int argc, char **argv)
{
    float acc,predict_time;
	parameters param;
	if (argc == 1)
	{
		printUsagePrompt(false, &param);
		return 0;
	}

	// vars
	char inputFileName[1024];
	char modelFileName[1024];
	char outputFileName[1024];
	vector <int> yLabels;
	vector <int> predLabels;
	vector <float> predScores;
	budgetedModel *model = NULL;
	FILE *pFile = NULL;

	// parse input string
	parseInputPrompt(argc, argv, false, inputFileName, modelFileName, outputFileName, &param);
	// for predict the only paremeter is whether the processing can be visualize so the parameter is only -v and the other\
	// information is about the input and the output file
	param.ALGORITHM = budgetedModel::getAlgorithm(modelFileName);

	// init random number generator
	srand((unsigned)time(NULL));

	// initialize test data and run trained model
	budgetedData *testData = NULL;


	delete testData;
	delete model;
	model = new budgetedModelLLSVM;
    if (!model->loadFromTextFile(modelFileName, &yLabels, &param))
    {
        printf("Error: can't read model from file %s.\n", modelFileName);
        delete model;
        return 1;
    }
    testData = new budgetedData(inputFileName, param.DIMENSION - (int) (param.BIAS_TERM != 0.0), param.CHUNK_SIZE, false, &yLabels);

    if (param.OUTPUT_SCORES)
        acc,predict_time = predictLLSVM(testData, &param, (budgetedModelLLSVM*) model, &predLabels, &predScores);
    else
        acc,predict_time = predictLLSVM(testData, &param, (budgetedModelLLSVM*) model, &predLabels);
	// print labels to output file
	pFile = fopen(outputFileName, "wt");
	if (!pFile)
	{
		printf("Error writing to output file %s.\n", outputFileName);
		return 1;
	}

	if (param.OUTPUT_SCORES)
	{
		for (unsigned int i = 0; i < predLabels.size(); i++)
			fprintf(pFile, "%d\t%f\n", predLabels[i], predScores[i]);
	}
	else
	{
		for (unsigned int i = 0; i < predLabels.size(); i++)
			fprintf(pFile, "%d\n", predLabels[i]);
	}
	fclose(pFile);
	return  acc,predict_time;
}}