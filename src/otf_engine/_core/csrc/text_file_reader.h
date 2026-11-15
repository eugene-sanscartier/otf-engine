/* -*- c++ -*- ----------------------------------------------------------
   Line reader for MTP potential files.
   Stands in for LAMMPS TextFileReader + ValueTokenizer, which the reference
   uses to parse the same format.
------------------------------------------------------------------------- */

#pragma once

#include <istream>
#include <sstream>
#include <string>

class TextFileReader {
  public:
    explicit TextFileReader(std::istream& is) : is_(is) {}

    // Advance to the next content line: blank lines and lines beginning with
    // '#' are skipped, and the separators "={}," plus any extra are replaced
    // by spaces. Returns false at end of file.
    bool next_line(const std::string& extra_separators = "");

    // First token of the current line.
    const std::string& keyword() const { return keyword_; }

    // The whole current line, with separators already replaced by spaces.
    const std::string& line() const { return line_; }

    // Tokens of the current line following the keyword.
    std::istringstream rest() const;

    // Advance one line and throw unless its keyword matches.
    void expect(const std::string& keyword);

    // Advance one line, check its keyword, and return the value after it.
    int next_int(const std::string& keyword);
    double next_double(const std::string& keyword);
    std::string next_string(const std::string& keyword);

    // Advance one line and throw at end of file, without checking the keyword.
    void advance();

  private:
    std::istream& is_;
    std::string line_;
    std::string keyword_;
};
