#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include "common_functions.h"
using namespace std;

int CommonTool::load_configure(const char* filename,map<string,string>& conf_dict)
{
	int cnt=0;
	ifstream fin(filename);
	if(!fin.is_open())
	{
		cout<<"open configure file:"<<filename<<" faild."<<endl;
		return -1;
	}
	string line;
	vector<string> result;
	conf_dict.clear();
	while(getline(fin,line))
	{
		if(line==string(""))	continue;
		if(string(line,0,1)==string("#"))   continue;
		result.clear();
		int n=split(line,'=',result);
		if(n!=2)
		{
			cout<<"Warning: broken configure item:"<<line<<endl;
		}
		conf_dict.insert(map<string,string>::value_type(result[0],result[1]));
	}
	return conf_dict.size();
}
string CommonTool::to_string(int x)
{
	ostringstream os;
	os<<x;
	return os.str();
}
int CommonTool::split(const string& str, char spliter,vector<string>& result)
{
 //   string s = "string, to, split";
	result.clear();
	if(str.size()==0)	return 0;
	istringstream ss( str );
	string feild;
	while (!ss.eof())
	{
		string x;			   // here's a nice, empty string
		getline( ss, feild, spliter );  // try to read the next field into it
		result.push_back(feild);
	   //   cout << x << endl;	  // print it out, even if we already hit EOF
	}
	return result.size();
}
string CommonTool::join(const vector<string>& strs,char sep)
{
	 if(strs.size()==0)
	 {
		return string("");
	 }
	 if(strs.size()==1)
	 {
		return strs[0];
	 }
	 string sep_str(1,sep);
	 string result;
	 for(int i=0;i<strs.size()-1;i++)
	 {
		 result+=strs[i]+sep_str;
	 }
	 return result+=strs.back();
}
double CommonTool::sigmod(double x)
{
	return 1.0/(1.0+exp(-x));
}
