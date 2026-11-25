/* -*- c++ -*- ----------------------------------------------------------
   Formatting of the fit log, which the fits and their line searches write.
------------------------------------------------------------------------- */

#pragma once

#include <cstdio>
#include <ostream>
#include <string>

// Returns the text that printf writes for format and args, up to 511 characters.
template <class... Args>
std::string format_text(const char* format, Args... args) {
    char text[512];
    if constexpr (sizeof...(Args) == 0)
        std::snprintf(text, sizeof text, "%s", format);
    else
        std::snprintf(text, sizeof text, format, args...);
    return text;
}

// Writes the line "stage: text" to log unless log is null, with text formatted from format and args.
template <class... Args>
void log_event(std::ostream* log, const char* stage, const char* format, Args... args) {
    if (log) *log << stage << ": " << format_text(format, args...) << std::endl;
}
