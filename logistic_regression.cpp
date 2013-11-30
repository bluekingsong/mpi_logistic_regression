#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <mpi.h>
#include "logistic_regression.h"
using namespace std;


int LogisticRegression::report(const lbfgsfloatval_t *x,const lbfgsfloatval_t *g,const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm,const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,int n,int k,int ls)
{
/*	if(k>=20 && k%this->save_intermediate_peroid==0 && this->intermediate_file_prefix.size()>0)
	{
		string parameter_values;
		this->get_parameter_values(parameter_values);
		ostringstream sout;
		sout<<this->intermediate_file_prefix<<"_bias"<<bias<<"_lamb"<<lambda<<"_iter"<<k;
		ofstream fout(sout.str().c_str());
		fout<<parameter_values;
		fout.close();
	}
*/
	lbfgsfloatval_t NLL=fx;
	this->train_credit.clear();
	for(size_t i=0;i<this->train_norm_size*this->mpi_size;++i)
	{
		if(this->recv_train_label[i]<0)	continue;
		this->train_credit.push_back(make_pair(recv_train_label[i],recv_train_ctr[i]));
	}
	double train_auc=this->evaluate_auc(this->train_credit);
	this->test_credit.clear();
	for(size_t i=0;i<this->test_norm_size*this->mpi_size;++i)
	{
		if(this->recv_test_label[i]<0)	continue;
		this->test_credit.push_back(make_pair(recv_test_label[i],recv_test_ctr[i]));
	}
	double test_auc=this->evaluate_auc(this->test_credit);
	double average_train_objective=fx/this->train_credit.size();
	double average_test_objective=this->test_objective_value/this->test_credit.size();
	ostringstream os;
	os<<"#iteration:"<<k<<"\ttrain_obj:"<<average_train_objective<<"\ttrain_auc:"<<train_auc<<"\ttest_obj:"<<average_test_objective<<"\ttest_auc:"<<test_auc<<endl;
	cerr<<os.str();
	cerr<<"end_iteration:"<<k<<"\ttrain objective:"<<fx<<"\ttest objective="<<test_objective_value<<endl<<flush;
	time_t t=time(0);
	cerr<<"current time:"<<asctime(localtime(&t))<<endl<<flush;
	if(k>=20)
	{
		if(this->test_objective_value-this->opt_test_objval>10)	return 1;
		if(this->test_objective_value<this->opt_test_objval)
		{
			this->opt_results=os.str();
			this->opt_test_objval=this->test_objective_value;
//			cerr<<"opt test obj="<<opt_test_objval<<endl;
		}
	}
	if(k>=100)	return 1;
	return 0;
}
lbfgsfloatval_t LogisticRegression::local_evaluate(const vector<string>& data,const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,float *ctr_vec)const
{
	memset(g,0,sizeof(lbfgsfloatval_t)*n);
	double obj_value=0;
	map<int,float> feature_vec;
	for(int line_cnt=0;line_cnt<data.size();++line_cnt)
	{
		const string& line=data[line_cnt];
		int yi=this->regularize(line,feature_vec);
		lbfgsfloatval_t ui=predict(feature_vec,w);
		lbfgsfloatval_t gradient=ui-yi;
		for(map<int,float>::const_iterator iter=feature_vec.begin();iter!=feature_vec.end();++iter)
		{
			int k=iter->first; //this->convert_index(iter->first,type);
			g[k]+=gradient*iter->second+this->lambda*w[k];
		}
		obj_value+=-log(ui*yi+(1-yi)*(1-ui)); // more robust
		ctr_vec[line_cnt]=ui;
	}
	return obj_value;
}
void LogisticRegression::lbfgs_slave()
{
	int n=this->feature2index.size();
	lbfgsfloatval_t *w=this->parameters;
	lbfgsfloatval_t *g=new lbfgsfloatval_t[n];
	while(true)
	{
		//cerr<<"###"<<mpi_rank<<": Bcast stopSign(slave need it)"<<endl;
		MPI_Bcast(&this->stopSign,1,MPI_INT,0,MPI_COMM_WORLD);
		if(this->stopSign)	break;
		do_work(w,g,n);
	}
	delete g;
}
pair<double,double> LogisticRegression::do_work(lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n)
{
	MPI_Bcast(w,n,MPI_FLOAT,0,MPI_COMM_WORLD); // broadcast the parameter w
	//cerr<<"##"<<this->mpi_rank<<": end broadcast w"<<endl;
	double test_obj=local_evaluate(test_data,w,g,n,test_ctr);
	memset(g,0,sizeof(lbfgsfloatval_t)*n);
	double train_obj=local_evaluate(train_data,w,g,n,train_ctr);
	if(this->mpi_rank==0)
	{
		MPI_Reduce(MPI_IN_PLACE,g,n,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE,&train_obj,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(MPI_IN_PLACE,&test_obj,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		//cerr<<"after reduce, train_obj="<<train_obj<<endl;
	}
	else
	{
		MPI_Reduce(g,0,n,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&train_obj,0,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		MPI_Reduce(&test_obj,0,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	}
	size_t size=this->train_norm_size;
	MPI_Gather(train_ctr,size,MPI_FLOAT,recv_train_ctr,size,MPI_FLOAT,0,MPI_COMM_WORLD);
	size=this->test_norm_size;
	MPI_Gather(test_ctr,size,MPI_FLOAT,recv_test_ctr,size,MPI_FLOAT,0,MPI_COMM_WORLD);
	return make_pair(train_obj,test_obj);
}
lbfgsfloatval_t LogisticRegression::evaluate(const lbfgsfloatval_t *w,lbfgsfloatval_t *g,const int n,const lbfgsfloatval_t step)
{
	//cerr<<"###"<<mpi_rank<<": Bcast stopSign(master give it)"<<endl;
	MPI_Bcast(&this->stopSign,1,MPI_INT,0,MPI_COMM_WORLD);
	pair<double,double> obj_pair=do_work(const_cast<lbfgsfloatval_t*>(w),g,n);
	// add regularized
	double para_reg=0;
	for(int i=0;i<n;++i)
	{
		para_reg+=this->lambda/2*w[i]*w[i];
	}
	obj_pair.first+=para_reg;
	obj_pair.second+=para_reg;
	this->test_objective_value=obj_pair.second;
	if(0.==step) // inital objective value
	{
		double train_obj=obj_pair.first;
		double test_obj=obj_pair.second;
		cerr<<"inital train objective value="<<train_obj<<" averaged="<<train_obj/this->train_norm_size/mpi_size<<endl;
		cerr<<"inital test objective value="<<test_obj<<" averaged="<<test_obj/this->test_norm_size/mpi_size<<endl<<flush;
	}
	return obj_pair.first;
}
int LogisticRegression::load_feature_map(const string& filename)
{
	ifstream fin(filename.c_str());
	this->feature2index.clear();
	int line_cnt=0;
	string line;
	vector<string> line_vec;
	while(getline(fin,line))
	{
		CommonTool::split(line,'\t',line_vec);
//		cerr<<line<<"##"<<line_vec[0]<<"##"<<line_vec[1]<<endl;
		int freq=atoi(line_vec[1].c_str());
		if(freq<300)	break;
		this->feature2index.insert(map<string,int>::value_type(line_vec[0],line_cnt));
		++line_cnt;
	}
	return line_cnt;
}
int LogisticRegression::optimize(const string& feature_configure,const string& intermediate_file_prefix,string& final_para_results,float lambda,float bias)
{
	int n=this->load_feature_map(feature_configure);
	if(this->mpi_rank==0)
	{
		cerr<<"read "<<n<<" features"<<endl;
		cerr<<"regularized parameter="<<lambda<<" bias="<<bias<<endl;
	}
	this->lambda=lambda;
	this->bias=bias;
	this->intermediate_file_prefix=intermediate_file_prefix;
	this->parameters=lbfgs_malloc(n);
	if(this->parameters==0)
	{
		cerr<<"malloc memery for parameters faild."<<endl;
		return -1;
	}
	double rmax=(double)RAND_MAX;
	memset(this->parameters,0,sizeof(lbfgsfloatval_t)*n);
	//config.linesearch=LBFGS_LINESEARCH_BACKTRACKING;
//	this->random_init_validation(test_output);
	int status=0;
	if(this->mpi_rank==0)  // master
	{
/*		this->opt_paras=lbfgs_malloc(n);
		if(this->opt_paras==0)
		{
			cerr<<"malloc memery for opt faild."<<endl;
			return -1;
		}
*/
		this->opt_test_objval=1e10;
		lbfgs_parameter_t config={
		40, 1e-5, 0, 1e-5,
		0, LBFGS_LINESEARCH_DEFAULT, 40,
		1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
		0.0, 0, -1,
		};
		lbfgsfloatval_t NLL=-1.0;
		time_t t=time(0);
		cerr<<"begin LBFGS, time:"<<asctime(localtime(&t))<<endl;
		status=lbfgs(n,this->parameters,&NLL,_evaluate,_reporter,this,&config);
		this->stopSign=1;
		//cerr<<"###"<<mpi_rank<<": Bcast stopSign(master give the last)"<<endl;
		MPI_Bcast(&this->stopSign,1,MPI_INT,0,MPI_COMM_WORLD);
		t=time(0);
		cerr<<"end LBFGS,status="<<status<<" NLL="<<NLL<<" time:"<<asctime(localtime(&t))<<endl;
		cout<<this->opt_results;
	}else{ // slave
		this->lbfgs_slave();
	}
	//get_parameter_values(final_para_results);
	lbfgs_free(this->parameters);
	return  status;
}
int LogisticRegression::get_parameter_values(string &result)const
{
	int n=this->feature2index.size();
	ostringstream sout;
	for(int i=0;i<n;i++)
	{
		sout<<this->parameters[i]<<endl;
	}
	result.assign(sout.str());
	return n;
}
int LogisticRegression::load_parameter_from_file(const char* filename)
{
	ifstream fin(filename);
	if(!fin)
	{
		cerr<<"Error, open "<<filename<<" failed."<<endl;
		return -1;
	}
	vector<string> lines;
	string line;
	while(getline(fin,line))
	{
		lines.push_back(line);
	}
	cerr<<"load "<<lines.size()<<" lines from "<<filename<<endl;
	this->parameters=new lbfgsfloatval_t[lines.size()];
	if(this->parameters==0)
	{
		cerr<<"Error, allocate memery faild."<<endl;
		return -1;
	}
	vector<string> field_vec;
	int n=lines.size();
	for(size_t i=0;i<lines.size();i++)
	{
		this->parameters[i]=atof(lines[i].c_str());
	}
	return lines.size();
}
double LogisticRegression::predict(const char *parameters_file,float bias,const char *test_file,const char *output)
{

	if(this->load_parameter_from_file(parameters_file)<0)
	{
		cerr<<"Error,load parameter file("<<parameters_file<<") failed."<<endl;
		return -1.0;
	}
	this->bias=bias;
	ifstream fin(test_file);
	if(!fin)
	{
		cerr<<"Error, open test file:"<<test_file<<" failed."<<endl;
		return -1.0;
	}
	ofstream fout(output);
	if(!fout)
	{
		cerr<<"Error,open output file:"<<output<<" failed."<<endl;
		return -1.0;
	}
	string line;
	vector<string> line_vec;
	map<int,float> feature_vec;
	int line_cnt=0;
	time_t t;
	while(getline(fin,line))
	{
		if(line_cnt%300000==0)
		{
			t=time(0);
			cerr<<"progress:"<<line_cnt<<" time:"<<asctime(localtime(&t))<<endl<<flush;
		}
		++line_cnt;
		int label=this->regularize(line,feature_vec);
		CommonTool::split(line,'\t',line_vec);
		lbfgsfloatval_t ui=predict(feature_vec,this->parameters);
		fout<<label<<'\t'<<ui<<endl;
	}
	fin.close();
	fout.close();
	if(this->parameters!=0)
		delete this->parameters;
	return 0;
}
double LogisticRegression::predict(const map<int,float>& feature_vec, const lbfgsfloatval_t *w)const
{
		double ti=this->bias;
		for(map<int,float>::const_iterator iter=feature_vec.begin();iter!=feature_vec.end();++iter)
		{
			ti+=w[iter->first]*(iter->second);
		}
		double ui=CommonTool::sigmod(ti);
		return ui;
}
int LogisticRegression::load_file2vec(const string& filename,vector<string>& vec,bool split)const
{
	if(this->mpi_size<=0 || this->mpi_rank<0)
	{
		cerr<<" illegal mpi_size("<<mpi_size<<") or illegal mpi_rank("<<mpi_rank<<")"<<endl;
		return 0;
	}
	ifstream fin(filename.c_str());
	if(!fin)
	{
		cerr<<"open file:"<<filename<<" faild."<<endl;
		return 0;
	}
	fin.seekg(0,fin.end);
	size_t file_size=fin.tellg();
	size_t guess_lines=file_size/4096; // assume 5K per line
	if(split)
		vec.reserve(guess_lines/mpi_size+10); // reserve capacity
	else
		vec.reserve(guess_lines+10);
	vec.clear();
	int line_cnt=0;
	fin.seekg(0,fin.beg);
	string line;
	while(getline(fin,line))
	{
		++line_cnt;
		if(split && (line_cnt-1)%this->mpi_size!=this->mpi_rank)	continue;
		vec.push_back(string());
		vec.back().swap(line);
	}
	if(this->mpi_rank==0)
	{
		time_t t=time(0);
		cerr<<"estimate total lines:"<<guess_lines<<" real total lines:"<<line_cnt<<endl;
		cerr<<"load "<<vec.size()<<" from file:"<<filename<<" at MPI rank:"<<mpi_rank<<" MPI size="<<mpi_size<<" time:"<<asctime(localtime(&t))<<endl<<flush;
	}
	fin.close();
	return line_cnt; //vec.size();
}
void LogisticRegression::init_label_data(size_t size,const vector<string>& data,int * &recv_buffer,vector<pair<int,float> >& credit_vec)const
{
	int *label_vec=new int[size];
	if(data.size()<size)	label_vec[size-1]=-1;
	vector<string> line_vec;
	for(size_t i=0;i<data.size();++i)
	{
		CommonTool::split(data[i],'\t',line_vec);
		label_vec[i]=atoi(line_vec[0].c_str());
	}
	if(this->mpi_rank==0)
	{
		recv_buffer=new int[size*this->mpi_size];
		credit_vec.reserve(size*this->mpi_size);
	}
	MPI_Gather(label_vec,size,MPI_INT,recv_buffer,size,MPI_INT,0,MPI_COMM_WORLD);
	delete []label_vec;
}
void LogisticRegression::init_ctr_data(size_t data_size,float *& ctr_vec,float *& recv_buffer)const
{
	ctr_vec=new float[data_size];
	if(this->mpi_rank==0)	recv_buffer=new float[data_size*this->mpi_size];
}
int LogisticRegression::init_data_buffer(const string& train_filename,const string& test_filename,bool split)
{
	//test data
	int lines=this->load_file2vec(test_filename,test_data,split);
	if(lines==0)	return -1;
	if(!split)
	{
		if(mpi_rank==0)	lines*=mpi_size;
		MPI_Bcast(&lines,1,MPI_INT,0,MPI_COMM_WORLD);
	}
	if(lines%this->mpi_size==0)	this->test_norm_size=lines/this->mpi_size;
	else	this->test_norm_size=lines/this->mpi_size+1;
	this->init_ctr_data(this->test_norm_size,this->test_ctr,this->recv_test_ctr);
	this->init_label_data(test_norm_size,this->test_data,this->recv_test_label,this->test_credit);
	//train data
	lines=this->load_file2vec(train_filename,train_data,split);
	if(lines==0)	return -1;
	if(!split)
	{
		if(mpi_rank==0)	lines*=mpi_size;
		MPI_Bcast(&lines,1,MPI_INT,0,MPI_COMM_WORLD);
	}
	if(lines%this->mpi_size==0)	this->train_norm_size=lines/this->mpi_size;
	else	this->train_norm_size=lines/this->mpi_size+1;
	this->init_ctr_data(train_norm_size,this->train_ctr,this->recv_train_ctr);
	this->init_label_data(train_norm_size,this->train_data,this->recv_train_label,this->train_credit);
	return 0;
}

int LogisticRegression::regularize(const vector<string>& line_vec,map<int,float>& feature_vec,int startIndex)const
{
	feature_vec.clear();
	for(int i=startIndex+1;i<line_vec.size();++i)
	{
		int index=this->map_feature2index(line_vec[i]);
		if(index>=0)
			feature_vec.insert(map<int,float>::value_type(index,1.0));
	}
	return atoi(line_vec[startIndex].c_str());
}
int LogisticRegression::regularize(const string& line,map<int,float>& feature_vec,int startIndex)const
{
	feature_vec.clear();
	vector<string> line_vec;
	CommonTool::split(line,'\t',line_vec);
	return regularize(line_vec,feature_vec,startIndex);
}
int LogisticRegression::map_feature2index(const string& name)const
{
	if(this->feature2index.find(name)==this->feature2index.end())
	{
//		cerr<<"could not find feature name!("<<name<<")"<<endl;
//		throw exception();
		return -1;
	}
	return this->feature2index.find(name)->second;
}

struct desc_sort_func{
	bool operator()(const pair<int,float>& left,const pair<int,float>& right)const
	{	return left.second<right.second;	}
};
double LogisticRegression::evaluate_auc(vector<pair<int,float> >& credit)
{
	if(credit.size()==0)	return 0;
	sort(credit.begin(),credit.end(),desc_sort_func());
	double n0=0,n1=0,s1=0;
	for(size_t i=0;i<credit.size();++i)
	{
		if(credit[i].first>0) // positive class
		{
			s1+=i+1;
			++n1;
		}else{
			++n0;
		}
	}
	return (s1-n1*(n1+1)/2.0)/(n0*n1);
}
LogisticRegression::LogisticRegression()
{
	this->save_intermediate_peroid=10;
	this->bias=0;
	this->lambda=0;
	this->mpi_rank=this->mpi_size=-1;
	this->stopSign=0;
	this->train_ctr=this->test_ctr=0;
	this->recv_train_label=this->recv_test_label=0;
	this->recv_train_ctr=this->recv_test_ctr=0;
}
LogisticRegression::~LogisticRegression()
{
	if(train_ctr!=0)	delete []train_ctr;
	if(test_ctr!=0)	delete []test_ctr;
	if(recv_train_label!=0)	delete []recv_train_label;
	if(recv_test_label!=0)	delete []recv_test_label;
	if(recv_train_ctr!=0)	delete []recv_train_ctr;
	if(recv_test_ctr!=0)	delete []recv_test_ctr;
}

