#pragma once
#include<vector>
#include "tiny_dnn\tiny_dnn.h"
#include "Common.h"

#define BATCH_SIZE 50
#define EPOCHS 3

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;



class TrainedNN
{
public:
	TrainedNN();
	TrainedNN(vector<vec_t>* inputs, vector<vec_t>* labels, vector<vec_t>* test_inputs, vector<vec_t>* test_labels, VEC_INT* core_nums);
	~TrainedNN();

	void Train();
	int Predict(int key);
	void Save(string path);

private:
	network<sequential> net;
	adam opt;
	vector<vec_t>* input_data;	
	vector<vec_t>* desired_out;
	vector<vec_t>* test_data;
	vector<vec_t>* test_target_data;
	int layer_nums;
	VEC_INT* cores;

};

