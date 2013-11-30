#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include <mpi.h>
#include "logistic_regression.h"
using namespace std;

int main(int argc,char **argv)
{
	if(argc!=6)
	{
		cerr<<"usage: ";
		cout<<argv[0]<<" train_feature_data test_feature_data regularization_parameter bias_parameter feature_config"<<endl;
		return -1;
	}
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
//	string train_data=string(argv[1])+string(".")+CommonTool::to_string(rank);
//	string test_data=string(argv[2])+string(".")+CommonTool::to_string(rank);
	bool split=true;
	string train_data=string(argv[1]);
	string test_data=string(argv[2]);
	LogisticRegression LR;
	LR.set_mpirank(rank);
	LR.set_mpisize(size);
	if(rank==0)
	{
		cerr<<" mpi rank="<<rank<<" mpi size="<<size<<endl;
		time_t t=time(0);
		cerr<<"start process:"<<asctime(localtime(&t))<<endl;
	}
//	LR.read_dataset("data/train.feature.filter","conf/overall.conf");
//	LR.init_data_buffer("data/train.feature.filter");
	LR.init_data_buffer(train_data,test_data,split);
	string parameters;
	float lambda=atof(argv[3]);
	float bias=atof(argv[4]);
	string feature_conf(argv[5]);
	int status=LR.optimize(feature_conf,string("model/para"),parameters,lambda,bias);
	if(rank==0)
	{
		cerr<<"LBFGS status:"<<status<<endl;
		time_t t=time(0);
		cerr<<"end process:"<<asctime(localtime(&t))<<endl;
	}
	MPI_Finalize();
	return 0;
}
