#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include "debug.h"

static const int max_stack_depth = 10;

#ifdef LINUX
void print_backtrace_linux (void * const *stack, int depth)
{
  char cmd[256];
  char proc_path[20];
  char exec_path[256];

  sprintf (proc_path, "/proc/%d/exe", getpid ());
  size_t length = readlink (proc_path, exec_path, 256);
  exec_path[length] = 0;

  for (int i = 0; i < depth; i++) {
    fprintf (stderr, "#%-3d ", i);
    fflush (stderr);
    auto addr = reinterpret_cast<size_t>(stack[i]);
    snprintf (cmd, sizeof(cmd), "addr2line -a -p -i -f -C -e %s 0x%zx 1>&2",
        exec_path, addr);
    if (system (cmd))
      fprintf (stderr, "   failed to exec: %s\n", cmd);
  }
}
#endif

#ifdef DARWIN
void print_backtrace_mac (void * const *stack, int depth)
{
  backtrace_symbols_fd (stack, depth, STDERR_FILENO);
}
#endif

void print_backtrace ()
{
  void *stack[max_stack_depth];
  size_t depth = backtrace (stack, max_stack_depth);

#ifdef LINUX
  print_backtrace_linux (stack, depth);
#endif

#ifdef DARWIN
  print_backtrace_mac (stack, depth);
#endif
}

void handler (int sig)
{
  fprintf (stderr, "error: signal %d\n", sig);
  print_backtrace ();
  exit (1);
}

typedef void (*sig_handler_func) (int);
sig_handler_func prev_sig_handler = signal (SIGSEGV, handler);
