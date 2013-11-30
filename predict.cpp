#include <ctime>
#include <iostream>
#include <fstream>
#include <string>
#include "logistic_regression.h"
using namespace std;

int main(int argc,char **argv)
{
	if(argc<4)
	{
		cout<<"usage:"<<argv[0]<<" parameter_file test_file output [lambda]"<<endl;
		return -1;
	}
	char *parameter_file=argv[1];
	char *test_file=argv[2];
	char *output=argv[3];
	float lambda=0;
	if(argc>=5)	lambda=atof(argv[4]);
	time_t t=time(0);
	cout<<"start process:"<<asctime(localtime(&t))<<endl;
	LogisticRegression LR;
	LR.load_feature_map(string("conf/feature.conf"));
	double NLL=LR.predict(parameter_file,lambda,test_file,output);
	cout<<"NLL for ("<<test_file<<") is "<<NLL<<endl;
	t=time(0);
	cout<<"end process:"<<asctime(localtime(&t))<<endl;
	return 0;
}
