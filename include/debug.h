#ifndef DEBUG_H_
#define DEBUG_H_

#include <cstdio>
#include <cassert>
#include <cstdarg>

#ifndef likely
#define likely(x)       __builtin_expect((x), 1)
#endif

#ifndef unlikely
#define unlikely(x)     __builtin_expect((x), 0)
#endif

#define __footprint \
  fprintf (stderr, "[footprint] %s:%d\n", __FILE__, __LINE__);

void print_backtrace ();

#define assert_bt(cond) \
if (unlikely (!(cond))) { \
    print_backtrace (); \
    assert (cond); \
};

#define assert_msg(cond, args...) \
if (unlikely (!(cond))) { \
    fprintf (stderr, args); \
    assert (cond); \
};

typedef enum {
  RESET = 0,
  BLACK,
  RED,
  GREEN,
  YELLOW,
  BLUE,
  MAGENTA,
  CYAN,
  WHITE
} COLOR;

#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_BLACK   "\x1b[30m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_WHITE   "\x1b[37m"

template <COLOR color>
int __attribute__((format(printf, 1, 2))) printf (const char * format, ...)
{
  static const char* colors[] = {
    ANSI_COLOR_RESET,
    ANSI_COLOR_BLACK,
    ANSI_COLOR_RED,
    ANSI_COLOR_GREEN,
    ANSI_COLOR_YELLOW,
    ANSI_COLOR_BLUE,
    ANSI_COLOR_MAGENTA,
    ANSI_COLOR_CYAN,
    ANSI_COLOR_WHITE
  };

  static class __Once {
    public: __Once () {
      setbuf(stdout, NULL);  // only disable buffer caching once
    }
  } __once;

  char buff[2048];
  va_list args;
  va_start (args, format);
  int count = vsnprintf (buff, sizeof(buff), format, args);
  va_end (args);
  fprintf (stderr, "%s%s%s", colors[color], buff, colors[0]);
  return count;
}

#define ERR  RED
#define WARN YELLOW
#define INFO BLUE
#define PASS GREEN

#endif /* DEBUG_H_ */
