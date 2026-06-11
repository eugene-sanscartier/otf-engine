/* -*- c++ -*- ----------------------------------------------------------
   Minimal stream-parsing utilities shared by the MTP file readers.
------------------------------------------------------------------------- */

#include "file_utils.h"

#include "mtp_potential.h"
#include "mtp_radial_basis.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// Strip leading/trailing whitespace from a string in-place.
static void trim(std::string& s) {
    size_t l = s.find_first_not_of(" \t\r\n");
    size_t r = s.find_last_not_of(" \t\r\n");
    s = (l == std::string::npos) ? "" : s.substr(l, r - l + 1);
}

// Replace every character in 'chars' found in 's' with a space.
static void replace_seps(std::string& s, const std::string& chars) {
    for (char& c : s)
        if (chars.find(c) != std::string::npos)
            c = ' ';
}

bool read_line(std::istream& is, std::string& out) {
    while (std::getline(is, out)) {
        trim(out);
        if (out.empty() || out[0] == '#')
            continue;
        replace_seps(out, "={},");
        return true;
    }
    return false;
}

bool read_line_dash(std::istream& is, std::string& out) {
    while (std::getline(is, out)) {
        trim(out);
        if (out.empty() || out[0] == '#')
            continue;
        replace_seps(out, "={},-");
        return true;
    }
    return false;
}

void read_basis_properties(std::istream& is, RadialMTPBasis& basis) {
    std::string line, kw;

    // Helper: read next line and extract first keyword token.
    auto read = [&]() {
        if (!read_line(is, line))
            throw std::runtime_error("MTP radial basis: unexpected end of file");
        std::istringstream(line) >> kw;
    };

    read();

    // Optional scaling line
    if (kw == "scaling") {
        std::istringstream(line) >> kw >> basis.scaling;
        read();
    }

    // Lower cutoff — accepts both 'min_dist' and 'min_val'
    if (kw != "min_val" && kw != "min_dist")
        throw std::runtime_error("MTP radial basis: expected min_dist, got '" + kw + "'");
    std::istringstream(line) >> kw >> basis.min_cutoff;

    // Upper cutoff — accepts both 'max_dist' and 'max_val'
    read();
    if (kw != "max_val" && kw != "max_dist")
        throw std::runtime_error("MTP radial basis: expected max_dist, got '" + kw + "'");
    std::istringstream(line) >> kw >> basis.max_cutoff;

    // Basis size
    read();
    if (kw != "radial_basis_size")
        throw std::runtime_error("MTP radial basis: expected radial_basis_size, got '" + kw + "'");
    std::istringstream(line) >> kw >> basis.size;

    // Allocate arrays
    basis.radial_basis_vals = new double[basis.size];
    basis.radial_basis_ders = new double[basis.size];
}

void read_file(const std::string& filename, MTPPotential& potential) {
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("MTPPotential: cannot open file '" + filename + "'");

    std::string line;

    // --- Header ---
    if (!read_line(f, line) || line.substr(0, 3) != "MTP")
        throw std::runtime_error("MTPPotential: file does not start with 'MTP'");

    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: missing version line");
    // After separator replacement '=' -> ' ', line is "version  1.1.0"
    {
        std::istringstream ss(line);
        std::string kw, ver;
        ss >> kw >> ver;
        if (ver != "1.1.0")
            throw std::runtime_error("MTPPotential: unsupported version '" + ver + "' (expected 1.1.0)");
    }

    // --- Optional and required top-level fields ---
    // We read one token-line at a time; some fields are optional so we need
    // one-line lookahead.
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected end of file");

    auto kw_of = [](const std::string& l) {
        std::string kw;
        std::istringstream ss(l);
        ss >> kw;
        return kw;
    };

    // potential_name (optional)
    if (kw_of(line) == "potential_name") {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        if (!(ss >> potential.potential_name))
            potential.potential_name = "";
        if (!read_line(f, line))
            throw std::runtime_error("MTPPotential: unexpected EOF");
    }

    // scaling (optional)
    if (kw_of(line) == "scaling") {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw >> potential.scaling;
        if (!read_line(f, line))
            throw std::runtime_error("MTPPotential: unexpected EOF");
    }

    // species_count (required)
    if (kw_of(line) != "species_count")
        throw std::runtime_error("MTPPotential: expected 'species_count', got '" + kw_of(line) + "'");
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw >> potential.species_count;
    }

    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");

    // potential_tag (optional)
    if (kw_of(line) == "potential_tag") {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        if (!(ss >> potential.potential_tag))
            potential.potential_tag = "";
        if (!read_line(f, line))
            throw std::runtime_error("MTPPotential: unexpected EOF");
    }

    // radial_basis_type (required)
    if (kw_of(line) != "radial_basis_type")
        throw std::runtime_error("MTPPotential: expected 'radial_basis_type', got '" + kw_of(line) + "'");
    {
        std::istringstream ss(line);
        std::string kw, basis_type;
        ss >> kw >> basis_type;
        if (basis_type == "RBChebyshev") {
            potential.radial_basis = new RBChebyshev(f); // reads min_dist, max_dist, radial_basis_size
        } else {
            throw std::runtime_error("MTPPotential: unsupported radial basis type '" + basis_type + "'");
        }
    }
    potential.radial_basis->scaling = potential.scaling;
    potential.radial_basis_size = potential.radial_basis->size;
    potential.min_cutoff = potential.radial_basis->min_cutoff;
    potential.max_cutoff = potential.radial_basis->max_cutoff;
    potential.max_cutoff_sq = potential.max_cutoff * potential.max_cutoff;

    // radial_funcs_count (immediately after radial basis block)
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "radial_funcs_count")
        throw std::runtime_error("MTPPotential: expected 'radial_funcs_count', got '" + kw_of(line) + "'");
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw >> potential.radial_func_count;
    }

    // radial_coeffs keyword
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) == "magnetic_basis_type")
        throw std::runtime_error("MTPPotential: magnetic basis is not supported");
    if (kw_of(line) != "radial_coeffs")
        throw std::runtime_error("MTPPotential: expected 'radial_coeffs', got '" + kw_of(line) + "'");

    // Allocate and read radial basis coefficients
    const int pairs_count = potential.species_count * potential.species_count;
    potential.radial_coeff_count_per_pair = potential.radial_basis_size * potential.radial_func_count;
    potential.radial_coeff_count = pairs_count * potential.radial_coeff_count_per_pair;
    potential.radial_basis_coeffs.resize(potential.radial_coeff_count);

    for (int i = 0; i < pairs_count; i++) {
        // pair-type label line "0-1" -> after dash-replacement: "0 1"
        if (!read_line_dash(f, line))
            throw std::runtime_error("MTPPotential: unexpected EOF in radial_coeffs");
        int type1, type2;
        {
            std::istringstream ss(line);
            ss >> type1 >> type2;
        }
        const int pair_offset = (type1 * potential.species_count + type2) * potential.radial_coeff_count_per_pair;

        for (int j = 0; j < potential.radial_func_count; j++) {
            if (!read_line(f, line))
                throw std::runtime_error("MTPPotential: unexpected EOF reading radial coefficients");
            std::istringstream ss(line);
            for (int k = 0; k < potential.radial_basis_size; k++) {
                if (!(ss >> potential.radial_basis_coeffs[pair_offset + j * potential.radial_basis_size + k]))
                    throw std::runtime_error("MTPPotential: not enough values in radial coefficient line");
            }
        }
    }

    // alpha_moments_count
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "alpha_moments_count")
        throw std::runtime_error("MTPPotential: expected 'alpha_moments_count'");
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw >> potential.alpha_moment_count;
    }
    potential.moment_tensor_vals.resize(potential.alpha_moment_count);
    potential.nbh_energy_ders_wrt_moments.resize(potential.alpha_moment_count);

    // alpha_index_basic_count
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "alpha_index_basic_count")
        throw std::runtime_error("MTPPotential: expected 'alpha_index_basic_count'");
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw >> potential.alpha_index_basic_count;
    }

    // alpha_index_basic — all values on one line after the keyword
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "alpha_index_basic")
        throw std::runtime_error("MTPPotential: expected 'alpha_index_basic'");
    potential.alpha_index_basic.resize(potential.alpha_index_basic_count);
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        int radial_func_max = 0;
        for (int i = 0; i < potential.alpha_index_basic_count; i++) {
            for (int j = 0; j < 4; j++) {
                if (!(ss >> potential.alpha_index_basic[i][j]))
                    throw std::runtime_error("MTPPotential: not enough values in alpha_index_basic");
            }
            if (potential.alpha_index_basic[i][0] > radial_func_max)
                radial_func_max = potential.alpha_index_basic[i][0];
        }
        if (radial_func_max != potential.radial_func_count - 1)
            throw std::runtime_error("MTPPotential: alpha_index_basic radial func index out of range");
    }

    // Compute max_alpha_index_basic (maximum rank among basic moments, +1)
    potential.max_alpha_index_basic = 0;
    for (int i = 0; i < potential.alpha_index_basic_count; i++) {
        int rank = potential.alpha_index_basic[i][1] + potential.alpha_index_basic[i][2] + potential.alpha_index_basic[i][3];
        if (rank > potential.max_alpha_index_basic)
            potential.max_alpha_index_basic = rank;
    }
    potential.max_alpha_index_basic++; // +1 for zeroth-order index

    // alpha_index_times_count
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "alpha_index_times_count")
        throw std::runtime_error("MTPPotential: expected 'alpha_index_times_count'");
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw >> potential.alpha_index_times_count;
    }

    // alpha_index_times — all values on one line after the keyword
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "alpha_index_times")
        throw std::runtime_error("MTPPotential: expected 'alpha_index_times'");
    potential.alpha_index_times.resize(potential.alpha_index_times_count);
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        for (int i = 0; i < potential.alpha_index_times_count; i++)
            for (int j = 0; j < 4; j++)
                if (!(ss >> potential.alpha_index_times[i][j]))
                    throw std::runtime_error("MTPPotential: not enough values in alpha_index_times");
    }

    // alpha_scalar_moments
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "alpha_scalar_moments")
        throw std::runtime_error("MTPPotential: expected 'alpha_scalar_moments'");
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw >> potential.alpha_scalar_count;
    }

    // alpha_moment_mapping
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "alpha_moment_mapping")
        throw std::runtime_error("MTPPotential: expected 'alpha_moment_mapping'");
    potential.alpha_moment_mapping.resize(potential.alpha_scalar_count);
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        for (int i = 0; i < potential.alpha_scalar_count; i++)
            if (!(ss >> potential.alpha_moment_mapping[i]))
                throw std::runtime_error("MTPPotential: not enough values in alpha_moment_mapping");
    }

    // species_coeffs
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "species_coeffs")
        throw std::runtime_error("MTPPotential: expected 'species_coeffs'");
    potential.species_coeffs.resize(potential.species_count);
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        for (int i = 0; i < potential.species_count; i++)
            if (!(ss >> potential.species_coeffs[i]))
                throw std::runtime_error("MTPPotential: not enough values in species_coeffs");
    }

    // moment_coeffs (linear_coeffs in our terminology)
    if (!read_line(f, line))
        throw std::runtime_error("MTPPotential: unexpected EOF");
    if (kw_of(line) != "moment_coeffs")
        throw std::runtime_error("MTPPotential: expected 'moment_coeffs'");
    potential.linear_coeffs.resize(potential.alpha_scalar_count);
    {
        std::istringstream ss(line);
        std::string kw;
        ss >> kw;
        for (int i = 0; i < potential.alpha_scalar_count; i++)
            if (!(ss >> potential.linear_coeffs[i]))
                throw std::runtime_error("MTPPotential: not enough values in moment_coeffs");
    }

    // Allocate working buffers (fixed size)
    potential.dist_powers.resize(potential.max_alpha_index_basic);
    potential.coord_powers.resize(potential.max_alpha_index_basic);
    potential.radial_vals.resize(potential.radial_func_count);
    potential.radial_ders.resize(potential.radial_func_count);

    // Zeroth-order power = 1
    potential.dist_powers[0] = potential.coord_powers[0][0] = potential.coord_powers[0][1] = potential.coord_powers[0][2] = 1.0;
}
