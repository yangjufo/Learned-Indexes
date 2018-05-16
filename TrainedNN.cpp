#include "TrainedNN.h"

TrainedNN::TrainedNN()
{
}

TrainedNN::TrainedNN(vector<vec_t>* inputs, vector<vec_t>* labels, vector<vec_t>* test_inputs, vector<vec_t>* test_labels, VEC_INT* core_nums)
{
	input_data = inputs;
	desired_out = labels;
	test_data = test_inputs;
	test_target_data = test_labels;
	cores = core_nums;	
	layer_nums = cores->size() - 1;
	opt.alpha = 0.0001;
	for (int i = 0; i < layer_nums; i++) {
		net << fc<relu>((*cores)[i], (*cores)[i + 1]);
	}
	net.weight_init(weight_init::constant(0.1));
	net.bias_init(weight_init::constant(0));
	Train();
}

TrainedNN::~TrainedNN()
{
}

void TrainedNN::Train()
{
	size_t batch_size = 1;
	size_t epochs = 1;
	net.fit<mse>(opt, *input_data, *desired_out, batch_size, epochs,
		[&]() {},
		[&]() {
	});
	/*net.fit<mse>(opt, input_data, desired_out, BATCH_SIZE, EPOCHS,
		[&]() {},
		[&]()
	{
		cout << Predict(1000) << endl;
	});*/

}

int TrainedNN::Predict(int key)
{
	vec_t res = net.predict(vec_t{ float(key) });
	return int(res[0] + 0.5);
}

void TrainedNN::Save(string path)
{
	net.save(path);
}



