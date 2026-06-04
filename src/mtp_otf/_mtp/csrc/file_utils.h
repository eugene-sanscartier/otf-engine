/* -*- c++ -*- ----------------------------------------------------------
   Minimal stream-parsing utilities shared by the MTP file readers.
------------------------------------------------------------------------- */

#pragma once

#include <istream>
#include <string>

class MTPPotential;
class RadialMTPBasis;

// Read the next non-blank, non-comment (#) line from 'is'.
// Returns false on EOF.  Separators (= { } ,) are replaced with spaces
// so the result can be fed directly to std::istringstream for tokenisation.
bool read_line(std::istream& is, std::string& out);

// Same as read_line but also replaces '-' (used in pair-type labels like "0-1").
bool read_line_dash(std::istream& is, std::string& out);

// Parse and fill radial basis metadata from stream.
void read_basis_properties(std::istream& is, RadialMTPBasis& basis);

// Parse and fill a full MTP potential object from file.
void read_file(const std::string& filename, MTPPotential& potential);
