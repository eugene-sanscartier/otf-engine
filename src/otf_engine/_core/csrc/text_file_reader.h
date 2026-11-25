/* -*- c++ -*- ----------------------------------------------------------
   Line reader for the MTP and ETN potential files.
------------------------------------------------------------------------- */

#pragma once

#include <istream>
#include <sstream>
#include <string>

class TextFileReader {
  public:
    explicit TextFileReader(std::istream& is) : is_(is) {}

    // Advances to the next line that is neither blank nor a '#' comment, and replaces the separators "={}," and
    // extra_separators by spaces. Returns false at end of file.
    bool next_line(const std::string& extra_separators = "");

    // First token of the current line.
    const std::string& keyword() const { return keyword_; }

    // The current line, with separators replaced by spaces.
    const std::string& line() const { return line_; }

    // Returns a stream of the tokens after the keyword.
    std::istringstream rest() const;

    // Advances one line and throws unless its keyword is keyword.
    void expect(const std::string& keyword);

    // Advances one line, checks its keyword, and returns the value after it.
    int next_int(const std::string& keyword);
    double next_double(const std::string& keyword);
    std::string next_string(const std::string& keyword);

    // Advances one line and throws at end of file.
    void advance();

  private:
    std::istream& is_;
    std::string line_;
    std::string keyword_;
};
