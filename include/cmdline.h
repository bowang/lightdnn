#ifndef CMDLINE_H_
#define CMDLINE_H_

#include <map>
#include <string>
#include <vector>

namespace lightdnn
{

typedef std::map<std::string, std::string> OptMap;

std::string vector2string (std::vector<unsigned> vec);

std::vector<unsigned> string2vector (std::string str);

OptMap parseCmdLine (int argc, char **argv);

}

#endif /* CMDLINE_H_ */
