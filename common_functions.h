#ifndef __COMMON_FUNCTIONS_H__
#define __COMMON_FUNCTIONS_H__

#include <map>
#include <vector>
#include <string>
using namespace std;

class CommonTool
{
  public:
	static int load_configure(const char* filename, map<string,string>& conf_dict);
	static int split(const string& str, char spliter,vector<string>& result);
	static string join(const vector<string>& strs,char sep);
	static double sigmod(double x);
	static string to_string(int x);
};
#endif
