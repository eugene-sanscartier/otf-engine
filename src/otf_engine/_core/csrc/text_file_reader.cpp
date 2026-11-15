/* -*- c++ -*- ----------------------------------------------------------
   Line reader for MTP potential files.
   Stands in for LAMMPS TextFileReader + ValueTokenizer, which the reference
   uses to parse the same format.
------------------------------------------------------------------------- */

#include "text_file_reader.h"

#include <stdexcept>

static const std::string SEPARATORS = "={},";

static void trim(std::string& s) {
    size_t l = s.find_first_not_of(" \t\r\n");
    size_t r = s.find_last_not_of(" \t\r\n");
    s = (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
}

bool TextFileReader::next_line(const std::string& extra_separators) {
    const std::string separators = SEPARATORS + extra_separators;

    while (std::getline(is_, line_)) {
        trim(line_);
        if (line_.empty() || line_[0] == '#')
            continue;
        for (char& c : line_)
            if (separators.find(c) != std::string::npos)
                c = ' ';
        keyword_.clear();
        std::istringstream(line_) >> keyword_;
        return true;
    }
    return false;
}

std::istringstream TextFileReader::rest() const {
    std::istringstream ss(line_);
    std::string kw;
    ss >> kw;
    return ss;
}

void TextFileReader::advance() {
    if (!next_line())
        throw std::runtime_error("MTP file: unexpected end of file");
}

void TextFileReader::expect(const std::string& keyword) {
    advance();
    if (keyword_ != keyword)
        throw std::runtime_error("MTP file: expected '" + keyword + "', got '" + keyword_ + "'");
}

int TextFileReader::next_int(const std::string& keyword) {
    expect(keyword);
    int value;
    if (!(rest() >> value))
        throw std::runtime_error("MTP file: no value after '" + keyword + "'");
    return value;
}

double TextFileReader::next_double(const std::string& keyword) {
    expect(keyword);
    double value;
    if (!(rest() >> value))
        throw std::runtime_error("MTP file: no value after '" + keyword + "'");
    return value;
}

std::string TextFileReader::next_string(const std::string& keyword) {
    expect(keyword);
    std::string value;
    rest() >> value; // optional fields may be empty
    return value;
}
