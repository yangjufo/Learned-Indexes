#include "Common.h"
#include "TrainedNN.h"
#include <fstream>
#include <string>
#include "tiny_dnn\tiny_dnn.h"
#include <ctime>
#include "btree.h"

#define TOTAL_NUM 1000000
#define BLOCK_SIZE 100

#define CLOCKS_PER_SEC  ((clock_t)1000)
typedef long clock_t;

int total_num;
enum DataDistribution
{
	Random = 0
};

vec_t ReadData(DataDistribution dataDis)
{
	vec_t data;
	string line;
	ifstream file;
	switch (dataDis) {
	case Random:
		file.open("random.csv");
		if (file.is_open()) {
			while (getline(file, line)) {
				data.push_back(atoi(line.c_str()));
			}
			file.close();
		}
		break;
	default:
		break;
	}
	return data;
}

 void HybridTraining(vector<vector<TrainedNN>> trainedIndex, int threshold, VEC_INT* stages, vector<VEC_INT>* core_nums, vector<vec_t>* train_set_x, vector<vec_t>* train_set_y, vector<vec_t>* test_set_x, vector<vec_t>* test_set_y)
{
	int stages_length = stages->size();
	vector<vector<vector<vec_t>>> tmp_inputs;
	vector<vector<vector<vec_t>>> tmp_labels;
	tmp_inputs.resize(stages_length);
	tmp_labels.resize(stages_length);
	tmp_inputs[0].push_back(vector<vec_t>{ *train_set_x });
	tmp_labels[0].push_back(vector<vec_t>{ *train_set_y });

	vector<vec_t> labels;
	float divisor;

	for (int i = 0; i < stages_length; i++) {
		for (int j = 0; j < (*stages)[i]; j++) {
			cout << i << " " << j << endl;
			labels.clear();					
			if (i == 0) {
				divisor = (*stages)[i + 1] * 1.0 / (TOTAL_NUM / BLOCK_SIZE);
				for (int k = 0; k < tmp_labels[i][j].size(); k++) {
					labels.push_back(vec_t{ float(int(tmp_labels[i][j][k][0] * divisor + 0.5)) });
				}
			}
			else {
				labels = tmp_labels[i][j];
			}
			trainedIndex[i][j] = TrainedNN(&tmp_inputs[i][j], &labels, test_set_x, test_set_y, &(*core_nums)[i]);
			if (i < stages_length - 1) {
				tmp_inputs[i + 1].resize((*stages)[i + 1]);
				tmp_labels[i + 1].resize((*stages)[i + 1]);
				for (int k = 0; k < tmp_inputs[i][j].size(); k++) {
					int pos = trainedIndex[i][j].Predict(tmp_inputs[i][j][k][0]);
					pos = pos > (*stages)[i + 1] - 1 ? (*stages)[i + 1] - 1 : pos;
					tmp_inputs[i + 1][pos].push_back(tmp_inputs[i][j][k]);
					tmp_labels[i + 1][pos].push_back(tmp_labels[i][j][k]);
				}
				tmp_inputs[0].clear();
				tmp_labels[0].clear();
			}		
		}
	}
}

int cmp(const void* a, const void* b)
{
	int ia = int(a);
	int ib = int(b);
	if (ia < ib)
		return -1;
	if (ia == ib)
		return 0;
	if (ia > ib)
		return 1;
}

int main()
{
	vec_t data = ReadData(Random);
	VEC_INT stages = { 1, 100 };
	int threshold = 0;
	vector<VEC_INT> core_nums = { {1, 1}, {1, 1} };
	vector<vec_t> train_x, train_y, test_x, test_y;		
	//train_x.resize(TOTAL_NUM);
	//train_y.resize(TOTAL_NUM);
	test_x.resize(TOTAL_NUM);
	test_y.resize(TOTAL_NUM);
	for (int i = 0; i < TOTAL_NUM; i++) {
		test_x[i].push_back(data[i]);
		test_y[i].push_back(int(i / BLOCK_SIZE));
		//train_x[i].push_back(data[i]);
		//train_y[i].push_back(int(i / BLOCK_SIZE));
	}
	cout << "Start train" << endl;
	vector<vector<TrainedNN>> trainedIndex;
	trainedIndex.resize(stages.size());
	trainedIndex[0].resize(1);
	trainedIndex[1].resize(stages[1]);
	HybridTraining(trainedIndex, threshold, &stages, &core_nums, &test_x, &test_y, &train_x, &train_y);
	double err = 0;
	int test_length = test_x.size();
	cout << "Calculate error" << endl;
	clock_t start_time = clock();
	for (int i = 0; i < test_length; i++) {
		int pre1 = trainedIndex[0][0].Predict(test_x[i][0]);
		pre1 = pre1 > stages[1] - 1 ? stages[1] - 1 : pre1;
		int pre2 = trainedIndex[1][pre1].Predict(test_x[i][0]);		
		err += abs(pre2 - test_y[i][0]);
	}
	clock_t end_time = clock();
	cout << "Average time: " << (end_time - start_time) * 1.0 / test_x.size() / CLOCKS_PER_SEC << endl;
	cout << "Average error: " << err / test_x.size() << endl;	

	cout << "Start build BTree" << endl;
	BTREE bt = btree_Create(sizeof(int), cmp);
	for (int i = 0; i < test_length; i++)
	{
		void* din;
		din = &test_x[i][0];
		btree_Insert(bt, din);
	}
	void *ret;
	void *key;
	start_time = clock();
	for (int i = 0; i < test_length; i++) {
		key = &test_x[i][0];
		btree_Search(bt, key, ret);
	}
	end_time = clock();
	cout << "Average time: " << (end_time - start_time) * 1.0 / test_x.size() / CLOCKS_PER_SEC << endl;
	system("pause");

	system("pause");
	return 0;
}