#ifndef __LR_H__
#define __LR_H__
#include <set>
#include <map>
#include <vector>
#include <string>
#include <sstream>
//#define LBFGS_FLOAT 32
#include <utility>
#include <lbfgs.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include "common_functions.h"
using namespace std;
class LogisticRegression
{
  private:
	lbfgsfloatval_t *parameters; // the parameters of logistic regression model we want to optimize
	//lbfgsfloatval_t *opt_paras;
	//int opt_iteration;
	double opt_test_objval;
	string opt_results;
	map<string,string> conf_dict; // configures
	//int lbfgs_iterations; // current lbfgs iterations
	int save_intermediate_peroid; // to save inter-mediate parameters, we set a peroid.
	string intermediate_file_prefix; // the prefix(include path) for intermediate result file.
	map<string,int> feature2index;
	float lambda; // regularization parameter
	float bias; // LR bias parameter;
	vector<pair<int,float> > train_credit;  // for master
	vector<pair<int,float> > test_credit; // for master
	double test_objective_value;
	///members for MPI
	vector<string> train_data;
	vector<string> test_data;
	int mpi_rank;
	int mpi_size;
	int stopSign;
	size_t train_norm_size;
	size_t test_norm_size;
	float *train_ctr;
	float *test_ctr;
	int *recv_train_label; // for master
	float *recv_train_ctr; // for master
	int *recv_test_label; // for master
	float *recv_test_ctr;  //for master
	///////// method
	int load_file2vec(const string& filename,vector<string>& vec,bool split)const;
	lbfgsfloatval_t local_evaluate(const vector<string>& data,const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,float *ctr_vec)const;
	pair<double,double> do_work(lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n);
	void lbfgs_slave();
	void init_ctr_data(size_t data_size,float *& ctr_vec,float *& recv_buffer)const;
	void init_label_data(size_t size,const vector<string>& data,int *&recv_buffer,vector<pair<int,float> >& credit_vec)const;
	static void init_parameters(lbfgsfloatval_t *parameter, unsigned int n); // init
	static lbfgsfloatval_t _evaluate(void *instance,const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,const lbfgsfloatval_t step)
	{
		return reinterpret_cast<LogisticRegression*>(instance)->evaluate(w,g,n,step);
	}
	static int _reporter(void *instance,const lbfgsfloatval_t *x,const lbfgsfloatval_t *g,const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,int n,int k,int ls)
	{
		return reinterpret_cast<LogisticRegression*>(instance)->report(x,g,fx,xnorm,gnorm,step,n,k,ls);
	}
	int report(const lbfgsfloatval_t *x,const lbfgsfloatval_t *g,const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,int n,int k,int ls);
	lbfgsfloatval_t evaluate(const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,const lbfgsfloatval_t step); // evaluate goal function and its gradients
	int map_feature2index(const string& name)const;
	int convert_index(int index,int type)const
	{	return (type-1)*feature2index.size()+index;		}
	void random_init_validation(const string& test_output,int folds=10); // random rerver record for validation and save it to test_output disk file

  public:
	//members for MPI
	void set_mpirank(int rank){ mpi_rank=rank;};
	void set_mpisize(int size){ mpi_size=size;};
	static double evaluate_auc(vector<pair<int,float> >& credit);
	LogisticRegression();
	~LogisticRegression();
	int load_feature_map(const string& filename); // load feature map,return the feature size
	int regularize(const string& line,map<int,float>& feature_vec,int startIndex=0)const; // extract feature vec, return the class label
	int regularize(const vector<string>& line_vec,map<int,float>& feature_vec,int startIndex=0)const; // extract feature vector, return class label.
	int read_dataset(const char* filename,const char* configure); // discard, use huge memory when data file is large
	int init_data_buffer(const string& train_filename,const string& test_filename,bool split=true); // read disk file to memory
	int prepare_read(); // init for read from data_buffer
	int finish_read(); // no use
//	istream& get_line(string &line);
	bool get_line(string &line); // get line from data_buffer
	int get_parameter_values(string &result)const;
	int optimize(const string& feature_configure,const string& intermediate_file_prefix,string& final_results,float lambda,float bias);
	int load_parameter_from_file(const char *filename);
	double predict(const map<int,float>& feature_vec,const lbfgsfloatval_t *w)const;
	double predict(const char *parameter_file,float bias,const char *test_file,const char *output);
};

#endif
