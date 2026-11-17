/* -*- c++ -*- ----------------------------------------------------------
   The event lines of a fit's log, which the fits and their line searches write.
------------------------------------------------------------------------- */

#pragma once

#include <cstdio>
#include <ostream>

// Writes "stage: text" to log, when there is one, with text formatted from format and args as printf formats them.
template <class... Args>
void log_event(std::ostream* log, const char* stage, const char* format, Args... args) {
    if (!log) return;
    char text[512];
    if constexpr (sizeof...(Args) == 0)
        std::snprintf(text, sizeof text, "%s", format);
    else
        std::snprintf(text, sizeof text, format, args...);
    *log << stage << ": " << text << std::endl;
}
