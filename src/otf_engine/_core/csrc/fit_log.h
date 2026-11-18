/* -*- c++ -*- ----------------------------------------------------------
   The text of a fit's log, which the fits and their line searches write.
------------------------------------------------------------------------- */

#pragma once

#include <cstdio>
#include <ostream>
#include <string>

// The text format and args give, as printf formats them.
template <class... Args>
std::string format_text(const char* format, Args... args) {
    char text[512];
    if constexpr (sizeof...(Args) == 0)
        std::snprintf(text, sizeof text, "%s", format);
    else
        std::snprintf(text, sizeof text, format, args...);
    return text;
}

// Writes the event line "stage: text" to log, when there is one, with text formatted from format and args.
template <class... Args>
void log_event(std::ostream* log, const char* stage, const char* format, Args... args) {
    if (log) *log << stage << ": " << format_text(format, args...) << std::endl;
}
