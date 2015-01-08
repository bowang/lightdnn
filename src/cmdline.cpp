#include <sstream>
#include <cstdlib>
#include "cmdline.h"

using namespace std;

namespace lightdnn
{

string vector2string (vector<unsigned> vec)
{
  if (vec.size () == 0) return "";
  char buffer[64];
  sprintf (buffer, "%d", vec[0]);
  string str (buffer);
  for (size_t i = 1; i < vec.size (); i++) {
    str += ",";
    sprintf (buffer, "%d", vec[i]);
    str += buffer;
  }
  return str;
}

vector<unsigned> string2vector (string str)
{
  vector<unsigned> vec;
  stringstream ss (str);
  string item;
  while (getline (ss, item, ',')) {
    vec.push_back (atoll (item.c_str ()));
  }
  return vec;
}

OptMap parseCmdLine (int argc, char **argv)
{
  OptMap options;

  int i = 0;
  while (i < argc) {
    char* key = argv[i];
    if (key[0] == '-') {
      char* arg = (i < argc - 1) ? argv[i + 1] : nullptr;
      string value (arg[0] != '-' ? arg : "true");
      i += (arg[0] != '-');
      options.emplace (key, value);
    }
    i++;
  }

  return options;
}

}
