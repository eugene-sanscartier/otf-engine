/* -*- c++ -*- ----------------------------------------------------------
   The moment tensor basis of an MTP, from its scalar basis functions.
   Ported from mlip-4 79a39bf, src/pots/mtp/mtp_basis.{h,cpp} (BSD 2-Clause):
   the split of each basis function into two and the elements it computes,
   written as PairMTP's alpha_index_basic, alpha_index_times and
   alpha_moment_mapping in place of mlip-4's program.
------------------------------------------------------------------------- */

#pragma once

#include <array>
#include <map>
#include <set>
#include <vector>

// A basis function: n radial indices and an n x n symmetric contraction matrix.
// Diagonal entry (i, i) is the tensor rank of moment i. Entry (i, j) is the number of indices moments i and j contract.
class MTP_Basis_Function {
  public:
    MTP_Basis_Function() = default;
    // elem holds the vector, then the matrix row by row: n + n^2 entries.
    explicit MTP_Basis_Function(std::vector<int> elem);

    bool operator<(const MTP_Basis_Function& b) const;

    int size() const { return size_; }
    void resize(int size) { size_ = size; elem_.resize((size_t) size_ * size_ + size_); }
    int& operator()(int i) { return elem_[i]; }
    int operator()(int i) const { return elem_[i]; }
    int& operator()(int i, int j) { return elem_[size_ + size_ * i + j]; }
    int operator()(int i, int j) const { return elem_[size_ + size_ * i + j]; }

    // Permutes the moments into canonical order, the least under operator<, and returns the permutation.
    std::vector<int> Canonize();
    // Returns the tensor rank, the sum of the diagonal entries minus the off-diagonal ones.
    int Dimension() const;

  protected:
    bool permutation_less(const std::vector<int>& p, const std::vector<int>& q) const;
    void permute(const std::vector<int>& p);

    int size_ = -1;
    std::vector<int> elem_;
};

// A basis function with its split into two sub-functions and the elements of its tensor.
class MTP_Basis_Function_Ext : public MTP_Basis_Function {
  public:
    MTP_Basis_Function_Ext(const MTP_Basis_Function& b) : MTP_Basis_Function(b) {}

    bool is_elementary() const { return size_ == 1; }

    // Chooses the split with the fewest operations, preferring sub-functions already in basis_functions.
    int BestOps(const std::set<MTP_Basis_Function_Ext>& basis_functions) const;
    // Builds the two canonized sub-functions of subset and sets perm, which maps their moments back.
    void ConstructSubBasisFunctions(MTP_Basis_Function sub_basis_function[2]) const;

    mutable std::vector<int> subset;       // 0 or 1 for each moment
    mutable int subset_size[2] = {0, 0};
    mutable std::vector<int> best_subset;
    // perm[0 .. subset_size[0]) are the moments of sub-function 0, the rest those of sub-function 1
    mutable std::vector<int> perm;
    mutable std::set<MTP_Basis_Function_Ext>::iterator from[2];
    mutable std::vector<int> element_inds;  // [3^Dimension()] element of each tensor component
};

// Moments and contractions of a set of scalar basis functions, in the form of the PairMTP members.
class MTP_Basis {
  public:
    // functions: the entries of each MTP_Basis_Function, a scalar with every index contracted.
    explicit MTP_Basis(const std::vector<std::vector<int>>& functions);

    int radial_function_count() const { return radial_function_count_; }

    std::vector<std::array<int, 4>> alpha_index_basic;  // the moments (mu, px, py, pz)
    std::vector<std::array<int, 4>> alpha_index_times;  // (i0, i1, mult, out)
    std::vector<int> alpha_moment_mapping;              // element of each scalar basis function
    int alpha_moment_count = 0;

  private:
    // Adds b and, recursively, the two sub-functions that compute it.
    std::set<MTP_Basis_Function_Ext>::iterator AddBasisFunction(const MTP_Basis_Function& b);
    void ProcessBasisFunctions();

    std::set<MTP_Basis_Function_Ext> basis_functions;
    int radial_function_count_ = 0;
    std::map<std::array<int, 4>, int> moment_to_id;
};

// Returns the basis functions of an MTP level, 6 to 28.
std::vector<std::vector<int>> level_basis_functions(int level);
